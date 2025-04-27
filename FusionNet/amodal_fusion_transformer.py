import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import lpips
import json
import yaml
from tqdm import tqdm
import argparse
import random
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import math

# 2D 위치 인코딩 클래스
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=256, max_w=256):
        """
        2D 위치 인코딩 초기화
        
        Args:
            d_model: 임베딩 차원
            max_h: 최대 높이
            max_w: 최대 너비
        """
        super().__init__()
        self.d_model = d_model
        
        # 사인 및 코사인 위치 인코딩 계산
        pe = torch.zeros(d_model, max_h, max_w)
        div_term = torch.exp(torch.arange(0, d_model//2, 2) * -(math.log(10000.0) / (d_model//2)))
        
        pos_h = torch.arange(0, max_h).unsqueeze(1).unsqueeze(1).float()
        pos_w = torch.arange(0, max_w).unsqueeze(0).unsqueeze(1).float()
        
        # 높이와 너비에 대한 인코딩 처리
        pe[0:d_model//2:2, :, :] = torch.sin(pos_h * div_term).transpose(1, 2)
        pe[1:d_model//2:2, :, :] = torch.cos(pos_h * div_term).transpose(1, 2)
        pe[d_model//2::2, :, :] = torch.sin(pos_w * div_term)
        pe[d_model//2+1::2, :, :] = torch.cos(pos_w * div_term)
        
        # 등록된 버퍼로 저장 (모델 상태의 일부로 저장됨, 파라미터는 아님)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        입력 텐서에 위치 인코딩 추가
        
        Args:
            x: 입력 텐서 [배치, 채널, 높이, 너비]
                
        Returns:
            위치 인코딩이 추가된 텐서
        """
        # 텐서 크기에 맞게 위치 인코딩 자르기
        _, _, h, w = x.size()
        pos_enc = self.pe[:, :h, :w]
        return x + pos_enc.unsqueeze(0)  # 배치 차원 추가


# Transformer 기반 AmodalFusionNet
class AmodalFusionTransformer(nn.Module):
    def __init__(self, in_channels, n_completions, d_model=256, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1):
        """
        Transformer 기반 Amodal Fusion Network 초기화
        
        Args:
            in_channels: 입력 채널 수 (3(부분 이미지) + 1(마스크) + 3*n_completions(각 완성 이미지))
            n_completions: amodal completion 이미지 개수
            d_model: Transformer 모델 차원
            nhead: 멀티헤드 어텐션의 헤드 수
            num_encoder_layers: 인코더 레이어 수
            num_decoder_layers: 디코더 레이어 수
            dim_feedforward: 피드포워드 네트워크의 히든 차원
            dropout: 드롭아웃 비율
        """
        super().__init__()
        self.n_completions = n_completions
        self.d_model = d_model
        
        # 이미지 특성 추출을 위한 초기 컨볼루션 레이어
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, d_model, kernel_size=1)
        )
        
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding2D(d_model)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True  # 배치가 첫 번째 차원으로 오도록 설정
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Transformer 디코더
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
        # 쿼리 임베딩 (디코더 시작점)
        self.query_embed = nn.Parameter(torch.randn(1, d_model))
        
        # 업샘플링을 위한 컨볼루션 레이어들
        self.upconv1 = nn.ConvTranspose2d(d_model, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 최종 출력 헤드 (n_completions 채널 생성)
        self.output_head = nn.Conv2d(64, n_completions, kernel_size=1)
        
    def forward(self, x):
        """
        순방향 전파
        
        Args:
            x: 입력 텐서 [배치, 채널, 높이, 너비]
                채널 = 3(부분 이미지) + 1(마스크) + 3*n_completions(완성 이미지들)
                
        Returns:
            각 픽셀별 가중치 맵 [배치, n_completions, 높이, 너비]
        """
        # 입력 크기 저장
        b, c, h, w = x.size()
        original_h, original_w = h, w
        
        # 초기 컨볼루션으로 특징 추출 및 다운샘플링
        # 입력: [B, C, H, W], 출력: [B, d_model, H/4, W/4]
        x = self.conv_encoder(x)
        
        # 위치 인코딩 추가
        x = self.pos_encoding(x)
        
        # 다운샘플링된 특징 맵 크기
        _, _, h_down, w_down = x.size()
        
        # Transformer 입력 형식으로 변환 [B, d_model, H/4, W/4] → [B, H/4*W/4, d_model]
        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, H*W, d_model]
        
        # Transformer 인코더 통과
        memory = self.transformer_encoder(x_flat)  # [B, H*W, d_model]
        
        # 쿼리 준비 (디코더 입력)
        query = self.query_embed.expand(b, -1, -1)  # [B, 1, d_model]
        
        # Transformer 디코더 통과 
        # 여기서는 단일 쿼리로 전체 특징을 디코딩
        # 더 복잡한 구현에서는 여러 쿼리를 사용할 수 있음
        decoder_output = self.transformer_decoder(query, memory)  # [B, 1, d_model]
        
        # 디코더 출력을 원래 특징 맵 형태로 융합
        # [B, 1, d_model] + [B, H*W, d_model] → [B, (H*W+1), d_model]
        combined = torch.cat([decoder_output, memory], dim=1)
        
        # 첫 번째(쿼리) 토큰을 제외하고 원래 공간 차원으로 재구성
        # [B, H*W, d_model] → [B, d_model, H/4, W/4]
        features = combined[:, 1:, :].permute(0, 2, 1).view(b, self.d_model, h_down, w_down)
        
        # 업샘플링을 통한 원래 해상도 복원
        x = F.relu(self.bn1(self.upconv1(features)))  # [B, 128, H/2, W/2]
        x = F.relu(self.bn2(self.upconv2(x)))  # [B, 64, H, W]
        
        # 최종 가중치 맵 생성
        weight_maps = self.output_head(x)  # [B, n_completions, H, W]
        
        # 소프트맥스로 가중치 정규화 (각 픽셀마다 합이 1)
        return F.softmax(weight_maps, dim=1)


# 데이터셋 클래스
class AmodalFusionDataset(Dataset):
    def __init__(self, root_dir, list_file, n_max_completions=5, transform=None):
        """
        Amodal Fusion 데이터셋 초기화
        
        Args:
            root_dir: 데이터셋의 루트 디렉토리
            list_file: 샘플 ID 리스트 파일 경로
            n_max_completions: 최대 completion 이미지 수 (고정된 채널 수를 위해)
            transform: 이미지 변환 함수
        """
        self.root_dir = Path(root_dir)
        with open(list_file, 'r') as f:
            self.ids = [line.strip() for line in f]
        self.n_max_completions = n_max_completions
        self.transform = transform
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        """데이터셋에서 하나의 샘플 가져오기"""
        img_id = self.ids[idx]
        
        # 원본 이미지, 마스크, 부분 이미지 로드
        X = Image.open(self.root_dir / f"{img_id}.jpg").convert("RGB")
        M = Image.open(self.root_dir / f"{img_id}_mask.png").convert("L")
        X_partial = Image.open(self.root_dir / f"{img_id}_partial.jpg").convert("RGB")
        
        # 모든 completion 이미지 로드
        completions = []
        for j in range(1, self.n_max_completions + 1):
            comp_path = self.root_dir / f"{img_id}_comp{j}.png"
            if comp_path.exists():
                comp = Image.open(comp_path).convert("RGB")
            else:
                # n_max_completions보다 적은 경우 검은색 이미지로 채움
                comp = Image.new("RGB", X.size, color=(0, 0, 0))
            completions.append(comp)
        
        # 변환 적용
        if self.transform:
            X = self.transform(X)
            X_partial = self.transform(X_partial)
            # 이진 마스크 변환을 위한 특별 처리
            if isinstance(M, Image.Image):
                M = torch.from_numpy(np.array(M)).float() / 255.0
                M = M.unsqueeze(0)  # [1, H, W] 형태로 만들기
            completions = [self.transform(c) for c in completions]
        
        # 완성 이미지들을 하나의 텐서로 스택
        completions_tensor = torch.stack(completions, dim=0)  # [N, 3, H, W]
        
        return X_partial, M, completions_tensor, X

# 훈련 함수
def train_fusion_transformer(args):
    """
    Transformer 기반 AmodalFusionNet 훈련 함수
    
    Args:
        args: 훈련 설정이 들어있는 argparse 객체
    """
    # 재현성을 위한 시드 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 텐서보드 로거 설정
    writer = SummaryWriter(args.log_dir)
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"장치: {device}")
    
    # 변환 설정
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 데이터셋 및 데이터로더 설정
    train_dataset = AmodalFusionDataset(
        args.data_dir,
        args.train_list,
        n_max_completions=args.n_max_completions,
        transform=transform
    )
    
    val_dataset = AmodalFusionDataset(
        args.data_dir,
        args.val_list,
        n_max_completions=args.n_max_completions,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 입력 채널 수 계산: 3(부분 이미지) + 1(마스크) + 3*n_completions(완성 이미지들)
    in_channels = 3 + 1 + 3 * args.n_max_completions
    
    # Transformer 기반 AmodalFusionNet 모델 초기화
    model = AmodalFusionTransformer(
        in_channels=in_channels, 
        n_completions=args.n_max_completions,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)
    
    # 모델 정보 출력
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # LPIPS 손실 초기화
    lpips_loss = lpips.LPIPS(net='vgg').to(device)
    
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 학습률 스케줄러 (옵션)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5, verbose=True
    )
    
    # 체크포인트 디렉토리 생성
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 손실 기록용
    best_val_loss = float('inf')
    start_epoch = 0
    
    # 체크포인트에서 이어서 훈련할 경우
    if args.resume and os.path.isfile(args.resume):
        print(f"체크포인트 로드 중: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"체크포인트 로드 완료 (에폭 {start_epoch})")
    
    # 훈련 루프
    for epoch in range(start_epoch, args.epochs):
        # 훈련 모드
        model.train()
        train_loss = 0.0
        train_l1_loss = 0.0
        train_lpips_loss = 0.0
        
        # 훈련 진행률 표시
        pbar = tqdm(train_loader, desc=f"에폭 {epoch+1}/{args.epochs} [훈련]")
        for batch_idx, (X_partial, M, completions, X) in enumerate(pbar):
            # GPU로 데이터 이동
            X_partial = X_partial.to(device)
            M = M.to(device)
            completions = completions.to(device)  # [B, N, 3, H, W]
            X = X.to(device)
            
            # [B, N, 3, H, W] → [B, N*3, H, W]로 변환
            B, N, C, H, W = completions.size()
            completions_flat = completions.view(B, N*C, H, W)
            
            # 입력 준비: 부분 이미지, 마스크, 완성 이미지들을 채널 방향으로 연결
            inputs = torch.cat([X_partial, M, completions_flat], dim=1)
            
            # AmodalFusionTransformer 순방향 전파
            weight_maps = model(inputs)  # [B, N, H, W]
            
            # 가중치를 이용해 완성 이미지들 융합
            weight_maps_expanded = weight_maps.unsqueeze(2)  # [B, N, 1, H, W]
            fused = (weight_maps_expanded * completions).sum(dim=1)  # [B, 3, H, W]
            
            # 융합된 이미지를 마스크 영역에만 적용하여 최종 입력 이미지 생성
            X_input = X_partial.clone()
            mask_expanded = M.expand_as(X_partial)
            X_input = X_input * (1 - mask_expanded) + fused * mask_expanded
            
            # 손실 계산 (마스크 영역에서만)
            # L1 손실
            l1_loss = F.l1_loss(X_input * mask_expanded, X * mask_expanded) 
            
            # LPIPS 손실 (지각적 유사성)
            # 마스크 영역만 비교하기 위해 mask_expanded로 곱해줌
            percept_loss = lpips_loss(X_input * mask_expanded, X * mask_expanded).mean()
            
            # 전체 손실
            loss = l1_loss + args.lpips_weight * percept_loss
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑 적용 (안정적인 훈련을 위해)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
            optimizer.step()
            
            # 손실 누적
            train_loss += loss.item()
            train_l1_loss += l1_loss.item()
            train_lpips_loss += percept_loss.item()
            
            # 진행률 표시 업데이트
            pbar.set_postfix({
                'loss': loss.item(),
                'l1': l1_loss.item(),
                'lpips': percept_loss.item()
            })
            
            # 일정 간격으로 샘플 시각화 저장 (옵션)
            if batch_idx % args.vis_interval == 0:
                save_visualization(
                    epoch, batch_idx, X_partial, M, completions, 
                    weight_maps, fused, X_input, X, args.vis_dir
                )
            
            # 중간 로깅
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % args.log_interval == 0:
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                writer.add_scalar('Train/L1_Loss', l1_loss.item(), global_step)
                writer.add_scalar('Train/LPIPS_Loss', percept_loss.item(), global_step)
        
        # 에폭 평균 손실
        avg_train_loss = train_loss / len(train_loader)
        avg_train_l1_loss = train_l1_loss / len(train_loader)
        avg_train_lpips_loss = train_lpips_loss / len(train_loader)
        
        print(f"에폭 {epoch+1} 훈련 손실: {avg_train_loss:.4f} (L1: {avg_train_l1_loss:.4f}, LPIPS: {avg_train_lpips_loss:.4f})")
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_l1_loss = 0.0
        val_lpips_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"에폭 {epoch+1}/{args.epochs} [검증]")
            for batch_idx, (X_partial, M, completions, X) in enumerate(pbar):
                # GPU로 데이터 이동
                X_partial = X_partial.to(device)
                M = M.to(device)
                completions = completions.to(device)
                X = X.to(device)
                
                # [B, N, 3, H, W] → [B, N*3, H, W]로 변환
                B, N, C, H, W = completions.size()
                completions_flat = completions.view(B, N*C, H, W)
                
                # 입력 준비
                inputs = torch.cat([X_partial, M, completions_flat], dim=1)
                
                # Transformer 순방향 전파
                weight_maps = model(inputs)
                
                # 융합
                weight_maps_expanded = weight_maps.unsqueeze(2)
                fused = (weight_maps_expanded * completions).sum(dim=1)
                
                # 마스크 영역에 적용
                X_input = X_partial.clone()
                mask_expanded = M.expand_as(X_partial)
                X_input = X_input * (1 - mask_expanded) + fused * mask_expanded
                
                # 손실 계산
                l1_loss = F.l1_loss(X_input * mask_expanded, X * mask_expanded)
                percept_loss = lpips_loss(X_input * mask_expanded, X * mask_expanded).mean()
                loss = l1_loss + args.lpips_weight * percept_loss
                
                # 손실 누적
                val_loss += loss.item()
                val_l1_loss += l1_loss.item()
                val_lpips_loss += percept_loss.item()
                
                # 진행률 표시 업데이트
                pbar.set_postfix({
                    'val_loss': loss.item(),
                    'val_l1': l1_loss.item(),
                    'val_lpips': percept_loss.item()
                })
                
                # 첫 배치의 이미지만 시각화
                if batch_idx == 0:
                    save_visualization(
                        epoch, 'val', X_partial, M, completions,
                        weight_maps, fused, X_input, X, args.vis_dir
                    )
        
        # 에폭 평균 검증 손실
        avg_val_loss = val_loss / len(val_loader)
        avg_val_l1_loss = val_l1_loss / len(val_loader)
        avg_val_lpips_loss = val_lpips_loss / len(val_loader)
        
        print(f"에폭 {epoch+1} 검증 손실: {avg_val_loss:.4f} (L1: {avg_val_l1_loss:.4f}, LPIPS: {avg_val_lpips_loss:.4f})")
        
        # 에폭 요약 로깅
        writer.add_scalar('Epoch/Train_Loss', avg_train_loss, epoch)
        writer.add_scalar('Epoch/Train_L1_Loss', avg_train_l1_loss, epoch)
        writer.add_scalar('Epoch/Train_LPIPS_Loss', avg_train_lpips_loss, epoch)
        writer.add_scalar('Epoch/Val_Loss', avg_val_loss, epoch)
        writer.add_scalar('Epoch/Val_L1_Loss', avg_val_l1_loss, epoch)
        writer.add_scalar('Epoch/Val_LPIPS_Loss', avg_val_lpips_loss, epoch)
        writer.add_scalar('Epoch/LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 학습률 조정
        scheduler.step(avg_val_loss)
        
        # 체크포인트 저장
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
        }
        
        # 일반 체크포인트 저장
        torch.save(checkpoint, f"{args.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")
        
        # 최고 성능 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, f"{args.checkpoint_dir}/best_model.pth")
            print(f"새로운 최고 모델 저장됨! 검증 손실: {best_val_loss:.4f}")
    
    writer.close()
    print(f"훈련 완료! 최고 검증 손실: {best_val_loss:.4f}")

def save_visualization(epoch, batch_idx, X_partial, M, completions, 
                      weight_maps, fused, X_input, X, output_dir):
    """
    훈련 및 검증 과정의 중간 결과물 시각화
    
    Args:
        epoch: 현재 에폭
        batch_idx: 배치 인덱스
        X_partial: 부분 이미지 [B, 3, H, W]
        M: 마스크 [B, 1, H, W]
        completions: 완성 이미지들 [B, N, 3, H, W]
        weight_maps: 가중치 맵 [B, N, H, W]
        fused: 융합된 이미지 [B, 3, H, W]
        X_input: 최종 입력 이미지 [B, 3, H, W]
        X: 원본 이미지 [B, 3, H, W]
        output_dir: 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 첫 번째 배치 이미지만 시각화
    b_idx = 0
    
    # 텐서를 CPU로 이동하고 numpy로 변환
    def convert_tensor(t):
        return t[b_idx].detach().cpu().numpy().transpose(1, 2, 0)
    
    # 시각화할 이미지 변환 ([-1, 1] → [0, 1])
    X_partial_vis = (convert_tensor(X_partial) + 1) / 2.0
    M_vis = convert_tensor(M).repeat(3, axis=2)  # 마스크를 RGB로 복제
    X_vis = (convert_tensor(X) + 1) / 2.0
    fused_vis = (convert_tensor(fused) + 1) / 2.0
    X_input_vis = (convert_tensor(X_input) + 1) / 2.0
    
    # 완성 이미지들 변환
    n_completions = completions.size(1)
    comp_vis = []
    for i in range(n_completions):
        comp = (convert_tensor(completions[:, i]) + 1) / 2.0
        comp_vis.append(comp)
    
    # 가중치 맵 변환
    weight_maps_vis = []
    for i in range(n_completions):
        w_map = convert_tensor(weight_maps[:, i].unsqueeze(1))
        weight_maps_vis.append(w_map.repeat(3, axis=2))  # 그레이스케일을 RGB로 복제
    
    # 이미지 그리드 생성
    fig, axs = plt.subplots(3, n_completions + 3, figsize=(3*(n_completions + 3), 9))
    
    # 첫 번째 행: 부분 이미지, 마스크, 원본 이미지, 완성 이미지들
    axs[0, 0].imshow(X_partial_vis)
    axs[0, 0].set_title('부분 이미지')
    axs[0, 1].imshow(M_vis, cmap='gray')
    axs[0, 1].set_title('마스크')
    axs[0, 2].imshow(X_vis)
    axs[0, 2].set_title('원본 이미지')
    
    for i in range(n_completions):
        axs[0, i+3].imshow(comp_vis[i])
        axs[0, i+3].set_title(f'완성 {i+1}')
    
    # 두 번째 행: 가중치 맵들
    for i in range(3):
        axs[1, i].axis('off')
    
    for i in range(n_completions):
        axs[1, i+3].imshow(weight_maps_vis[i])
        axs[1, i+3].set_title(f'가중치 맵 {i+1}')
    
    # 세 번째 행: 융합된 이미지, 최종 입력 이미지, 원본 이미지 비교
    axs[2, 0].imshow(fused_vis)
    axs[2, 0].set_title('융합된 이미지')
    axs[2, 1].imshow(X_input_vis)
    axs[2, 1].set_title('최종 입력')
    axs[2, 2].imshow(X_vis)
    axs[2, 2].set_title('원본 (타겟)')
    
    for i in range(n_completions):
        axs[2, i+3].axis('off')
    
    # 모든 축의 축 제거
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 시각화 저장
    plt.tight_layout()
    plt.savefig(f"{output_dir}/vis_epoch_{epoch}_{batch_idx}.png")
    plt.close()

def train_distributed(local_rank, world_size, args):
    """
    분산 훈련을 위한 함수
    
    Args:
        local_rank: 로컬 GPU 순위
        world_size: 총 GPU 수
        args: 훈련 설정이 들어있는 argparse 객체
    """
    # 분산 환경 설정
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=local_rank
    )
    
    # 장치 설정
    device = torch.device(f'cuda:{local_rank}')
    
    # 재현성을 위한 시드 설정 (각 프로세스마다 다른 시드 사용)
    seed = args.seed + local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 텐서보드 로거 설정 (마스터 프로세스만)
    writer = None
    if local_rank == 0:
        writer = SummaryWriter(args.log_dir)
        
        # 출력 디렉토리 생성
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.vis_dir, exist_ok=True)
        
        print(f"총 {world_size}개의 GPU를 사용하여 분산 훈련을 시작합니다.")
    
    # 변환 설정
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 데이터셋 및 분산 데이터로더 설정
    train_dataset = AmodalFusionDataset(
        args.data_dir,
        args.train_list,
        n_max_completions=args.n_max_completions,
        transform=transform
    )
    
    val_dataset = AmodalFusionDataset(
        args.data_dir,
        args.val_list,
        n_max_completions=args.n_max_completions,
        transform=transform
    )
    
    # 분산 샘플러 사용
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 분산 샘플러에서 셔플 처리
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    
    # 검증은 모든 프로세스에서 동일한 데이터 평가를 위해 샘플러 없이 사용
    # 하지만 각 프로세스는 일부만 처리 (결과는 나중에 수집)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 입력 채널 수 계산: 3(부분 이미지) + 1(마스크) + 3*n_completions(완성 이미지들)
    in_channels = 3 + 1 + 3 * args.n_max_completions
    
    # Transformer 기반 AmodalFusionNet 모델 초기화
    model = AmodalFusionTransformer(
        in_channels=in_channels, 
        n_completions=args.n_max_completions,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)
    
    # 분산 데이터 병렬 모델로 래핑
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True  # 모델에 사용되지 않는 파라미터가 있을 경우
    )
    
    # 마스터 프로세스에서만 모델 정보 출력
    if local_rank == 0:
        print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # LPIPS 손실 초기화
    lpips_loss = lpips.LPIPS(net='vgg').to(device)
    
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 학습률 스케줄러 (옵션)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5, verbose=(local_rank == 0)
    )
    
    # 손실 기록용
    best_val_loss = float('inf')
    start_epoch = 0
    
    # 체크포인트에서 이어서 훈련할 경우
    if args.resume and os.path.isfile(args.resume):
        # 모든 프로세스가 같은 위치에서 모델을 로드
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        # DDP 모델의 경우 '.module' 접두사가 없는 상태 딕셔너리 필요
        if 'module' in list(checkpoint['model_state_dict'].keys())[0] and not hasattr(model, 'module'):
            # 이미 'module.' 접두사가 있는 경우 제거
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                new_state_dict[k.replace('module.', '')] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if local_rank == 0:
            print(f"체크포인트 로드 완료 (에폭 {start_epoch})")
    
    # 훈련 루프
    for epoch in range(start_epoch, args.epochs):
        # 에폭 시작 시 샘플러 설정
        train_sampler.set_epoch(epoch)
        
        # 훈련 모드
        model.train()
        train_loss = 0.0
        train_l1_loss = 0.0
        train_lpips_loss = 0.0
        
        # 훈련 진행률 표시 (마스터 프로세스만)
        if local_rank == 0:
            pbar = tqdm(train_loader, desc=f"에폭 {epoch+1}/{args.epochs} [훈련]")
        else:
            pbar = train_loader
            
        for batch_idx, (X_partial, M, completions, X) in enumerate(pbar):
            # GPU로 데이터 이동
            X_partial = X_partial.to(device)
            M = M.to(device)
            completions = completions.to(device)  # [B, N, 3, H, W]
            X = X.to(device)
            
            # [B, N, 3, H, W] → [B, N*3, H, W]로 변환
            B, N, C, H, W = completions.size()
            completions_flat = completions.view(B, N*C, H, W)
            
            # 입력 준비: 부분 이미지, 마스크, 완성 이미지들을 채널 방향으로 연결
            inputs = torch.cat([X_partial, M, completions_flat], dim=1)
            
            # AmodalFusionTransformer 순방향 전파
            weight_maps = model(inputs)  # [B, N, H, W]
            
            # 가중치를 이용해 완성 이미지들 융합
            weight_maps_expanded = weight_maps.unsqueeze(2)  # [B, N, 1, H, W]
            fused = (weight_maps_expanded * completions).sum(dim=1)  # [B, 3, H, W]
            
            # 융합된 이미지를 마스크 영역에만 적용하여 최종 입력 이미지 생성
            X_input = X_partial.clone()
            mask_expanded = M.expand_as(X_partial)
            X_input = X_input * (1 - mask_expanded) + fused * mask_expanded
            
            # 손실 계산 (마스크 영역에서만)
            # L1 손실
            l1_loss = F.l1_loss(X_input * mask_expanded, X * mask_expanded) 
            
            # LPIPS 손실 (지각적 유사성)
            # 마스크 영역만 비교하기 위해 mask_expanded로 곱해줌
            percept_loss = lpips_loss(X_input * mask_expanded, X * mask_expanded).mean()
            
            # 전체 손실
            loss = l1_loss + args.lpips_weight * percept_loss
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑 적용 (안정적인 훈련을 위해)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
            optimizer.step()
            
            # 손실 누적
            train_loss += loss.item()
            train_l1_loss += l1_loss.item()
            train_lpips_loss += percept_loss.item()
            
            # 진행률 표시 업데이트 (마스터 프로세스만)
            if local_rank == 0:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'l1': l1_loss.item(),
                    'lpips': percept_loss.item()
                })
                
                # 일정 간격으로 샘플 시각화 저장 (옵션)
                if batch_idx % args.vis_interval == 0:
                    save_visualization(
                        epoch, batch_idx, X_partial, M, completions, 
                        weight_maps, fused, X_input, X, args.vis_dir
                    )
                
                # 중간 로깅
                global_step = epoch * len(train_loader) + batch_idx
                if batch_idx % args.log_interval == 0:
                    writer.add_scalar('Train/Loss', loss.item(), global_step)
                    writer.add_scalar('Train/L1_Loss', l1_loss.item(), global_step)
                    writer.add_scalar('Train/LPIPS_Loss', percept_loss.item(), global_step)
        
        # 손실값 수집을 위한 텐서 생성
        train_loss_tensor = torch.tensor([train_loss, train_l1_loss, train_lpips_loss], device=device)
        
        # 모든 프로세스의 손실값 수집
        torch.distributed.all_reduce(train_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        
        # 평균 계산 (모든 프로세스의 합을 프로세스 수와 데이터 수로 나눔)
        train_loss_tensor /= (world_size * len(train_loader))
        
        # 에폭 평균 손실
        avg_train_loss = train_loss_tensor[0].item()
        avg_train_l1_loss = train_loss_tensor[1].item()
        avg_train_lpips_loss = train_loss_tensor[2].item()
        
        if local_rank == 0:
            print(f"에폭 {epoch+1} 훈련 손실: {avg_train_loss:.4f} (L1: {avg_train_l1_loss:.4f}, LPIPS: {avg_train_lpips_loss:.4f})")
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_l1_loss = 0.0
        val_lpips_loss = 0.0
        
        with torch.no_grad():
            if local_rank == 0:
                pbar = tqdm(val_loader, desc=f"에폭 {epoch+1}/{args.epochs} [검증]")
            else:
                pbar = val_loader
                
            for batch_idx, (X_partial, M, completions, X) in enumerate(pbar):
                # GPU로 데이터 이동
                X_partial = X_partial.to(device)
                M = M.to(device)
                completions = completions.to(device)
                X = X.to(device)
                
                # [B, N, 3, H, W] → [B, N*3, H, W]로 변환
                B, N, C, H, W = completions.size()
                completions_flat = completions.view(B, N*C, H, W)
                
                # 입력 준비
                inputs = torch.cat([X_partial, M, completions_flat], dim=1)
                
                # Transformer 순방향 전파
                weight_maps = model(inputs)
                
                # 융합
                weight_maps_expanded = weight_maps.unsqueeze(2)
                fused = (weight_maps_expanded * completions).sum(dim=1)
                
                # 마스크 영역에 적용
                X_input = X_partial.clone()
                mask_expanded = M.expand_as(X_partial)
                X_input = X_input * (1 - mask_expanded) + fused * mask_expanded
                
                # 손실 계산
                l1_loss = F.l1_loss(X_input * mask_expanded, X * mask_expanded)
                percept_loss = lpips_loss(X_input * mask_expanded, X * mask_expanded).mean()
                loss = l1_loss + args.lpips_weight * percept_loss
                
                # 손실 누적
                val_loss += loss.item()
                val_l1_loss += l1_loss.item()
                val_lpips_loss += percept_loss.item()
                
                # 진행률 표시 업데이트 (마스터 프로세스만)
                if local_rank == 0:
                    pbar.set_postfix({
                        'val_loss': loss.item(),
                        'val_l1': l1_loss.item(),
                        'val_lpips': percept_loss.item()
                    })
                    
                    # 첫 배치의 이미지만 시각화
                    if batch_idx == 0:
                        save_visualization(
                            epoch, 'val', X_partial, M, completions,
                            weight_maps, fused, X_input, X, args.vis_dir
                        )
        
        # 검증 손실값 수집을 위한 텐서 생성
        val_loss_tensor = torch.tensor([val_loss, val_l1_loss, val_lpips_loss], device=device)
        # 샘플 수 추적을 위한 텐서
        val_count_tensor = torch.tensor([len(val_loader)], device=device)
        
        # 모든 프로세스의 손실값 수집
        torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_count_tensor, op=torch.distributed.ReduceOp.SUM)
        
        # 평균 계산 (모든 프로세스의 합을 총 데이터 수로 나눔)
        val_loss_tensor /= val_count_tensor.item()
        
        # 에폭 평균 검증 손실
        avg_val_loss = val_loss_tensor[0].item()
        avg_val_l1_loss = val_loss_tensor[1].item()
        avg_val_lpips_loss = val_loss_tensor[2].item()
        
        if local_rank == 0:
            print(f"에폭 {epoch+1} 검증 손실: {avg_val_loss:.4f} (L1: {avg_val_l1_loss:.4f}, LPIPS: {avg_val_lpips_loss:.4f})")
            
            # 에폭 요약 로깅
            writer.add_scalar('Epoch/Train_Loss', avg_train_loss, epoch)
            writer.add_scalar('Epoch/Train_L1_Loss', avg_train_l1_loss, epoch)
            writer.add_scalar('Epoch/Train_LPIPS_Loss', avg_train_lpips_loss, epoch)
            writer.add_scalar('Epoch/Val_Loss', avg_val_loss, epoch)
            writer.add_scalar('Epoch/Val_L1_Loss', avg_val_l1_loss, epoch)
            writer.add_scalar('Epoch/Val_LPIPS_Loss', avg_val_lpips_loss, epoch)
            writer.add_scalar('Epoch/LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 학습률 조정 (모든 프로세스 동일하게)
        scheduler.step(avg_val_loss)
        
        # 체크포인트 저장 (마스터 프로세스만)
        if local_rank == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }
            
            # 일반 체크포인트 저장
            torch.save(checkpoint, f"{args.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")
            
            # 최고 성능 모델 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(checkpoint, f"{args.checkpoint_dir}/best_model.pth")
                print(f"새로운 최고 모델 저장됨! 검증 손실: {best_val_loss:.4f}")
    
    # 훈련 종료
    if local_rank == 0:
        writer.close()
        print(f"훈련 완료! 최고 검증 손실: {best_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Transformer 기반 AmodalFusionNet 훈련을 위한 스크립트')
    
    # 데이터 관련 인수
    parser.add_argument('--data_dir', type=str, required=True, help='데이터셋 디렉토리')
    parser.add_argument('--train_list', type=str, required=True, help='훈련 샘플 ID 목록 파일')
    parser.add_argument('--val_list', type=str, required=True, help='검증 샘플 ID 목록 파일')
    parser.add_argument('--img_size', type=int, default=256, help='이미지 크기')
    parser.add_argument('--n_max_completions', type=int, default=5, help='최대 완성 이미지 수')
    
    # 모델 관련 인수
    parser.add_argument('--d_model', type=int, default=256, help='Transformer 모델 차원')
    parser.add_argument('--nhead', type=int, default=8, help='멀티헤드 어텐션의 헤드 수')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='인코더 레이어 수')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='디코더 레이어 수')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='피드포워드 네트워크의 히든 차원')
    parser.add_argument('--dropout', type=float, default=0.1, help='드롭아웃 비율')
    
    # 훈련 관련 인수
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=50, help='에폭 수')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--lpips_weight', type=float, default=1.0, help='LPIPS 손실 가중치')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='그래디언트 클리핑 값 (0이면 비활성화)')
    parser.add_argument('--num_workers', type=int, default=4, help='데이터 로더 워커 수')
    parser.add_argument('--seed', type=int, default=42, help='난수 시드')
    parser.add_argument('--resume', type=str, help='체크포인트에서 이어서 훈련')
    parser.add_argument('--cpu', action='store_true', help='CPU 사용 (디버깅용)')
    
    # 분산 훈련 관련 인수
    parser.add_argument('--nodes', type=int, default=1, help='사용할 노드 수')
    parser.add_argument('--gpus', type=int, default=1, help='노드당 GPU 수')
    parser.add_argument('--node_rank', type=int, default=0, help='현재 노드 순위')
    
    # 출력 및 로깅 관련 인수
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='체크포인트 저장 디렉토리')
    parser.add_argument('--log_dir', type=str, default='logs', help='텐서보드 로그 디렉토리')
    parser.add_argument('--vis_dir', type=str, default='visualizations', help='시각화 저장 디렉토리')
    parser.add_argument('--log_interval', type=int, default=10, help='로깅 간격 (배치 단위)')
    parser.add_argument('--vis_interval', type=int, default=100, help='시각화 간격 (배치 단위)')
    
    # 설정 파일을 통한 인수 지정 가능
    parser.add_argument('--config', type=str, help='YAML 설정 파일')
    
    args = parser.parse_args()
    
    # 설정 파일을 통한 인수 로드 (제공된 경우)
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)
    
    # 단일 GPU 또는 CPU 모드
    if args.cpu or args.gpus <= 1:
        # 출력 디렉토리 생성
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.vis_dir, exist_ok=True)
        
        # 설정 저장 (재현을 위해)
        config_path = os.path.join(args.checkpoint_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(vars(args), f)
        
        # 비분산 훈련 시작
        train_fusion_transformer(args)
    else:
        # 멀티 GPU 분산 훈련
        # 환경변수로 분산 설정 전달
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # 총 프로세스 수 계산
        world_size = args.gpus
        
        # 마스터 프로세스에서만 출력 디렉토리 생성
        if args.node_rank == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            os.makedirs(args.log_dir, exist_ok=True)
            os.makedirs(args.vis_dir, exist_ok=True)
            
            # 설정 저장 (재현을 위해)
            config_path = os.path.join(args.checkpoint_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(vars(args), f)
        
        # 멀티프로세싱 시작 (GPU당 하나의 프로세스)
        import torch.multiprocessing as mp
        mp.spawn(
            train_distributed,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )

if __name__ == '__main__':
    main()