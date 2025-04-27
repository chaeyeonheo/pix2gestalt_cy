import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
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

def load_image(path, transform=None):
    """이미지 로드 및 변환"""
    img = Image.open(path).convert('RGB')
    if transform:
        img = transform(img)
    return img

def load_mask(path, img_size):
    """마스크 로드 및 텐서로 변환"""
    mask = Image.open(path).convert('L')
    mask = mask.resize((img_size, img_size), Image.NEAREST)
    mask = np.array(mask) / 255.0
    mask = torch.from_numpy(mask).float().unsqueeze(0)  # [1, H, W]
    return mask

def process_image(
    image_id, 
    data_dir, 
    model, 
    img_size=256,
    n_max_completions=5,
    device='cuda',
    output_dir='outputs'
):
    """
    단일 이미지에 대한 처리 및 결과 저장
    
    Args:
        image_id: 이미지 ID (파일 이름의 기본)
        data_dir: 데이터 디렉토리
        model: Transformer 모델
        img_size: 이미지 크기
        n_max_completions: 최대 완성 이미지 수
        device: 텐서를 처리할 장치
        output_dir: 결과 저장 디렉토리
    
    Returns:
        완성된 입력 이미지
    """
    # 변환 설정
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 이미지 경로 생성
    img_path = os.path.join(data_dir, f"{image_id}.jpg")
    mask_path = os.path.join(data_dir, f"{image_id}_mask.png")
    partial_path = os.path.join(data_dir, f"{image_id}_partial.jpg")
    
    # 이미지 및 마스크 로드
    X = load_image(img_path, transform)
    M = load_mask(mask_path, img_size)
    X_partial = load_image(partial_path, transform)
    
    # 완성 이미지 로드
    completions = []
    for j in range(1, n_max_completions + 1):
        comp_path = os.path.join(data_dir, f"{image_id}_comp{j}.png")
        if os.path.exists(comp_path):
            comp = load_image(comp_path, transform)
        else:
            # n_max_completions보다 적은 경우 검은색 이미지로 채움
            comp = torch.zeros_like(X)
        completions.append(comp)
    
    # 배치 차원 추가
    X = X.unsqueeze(0).to(device)
    M = M.unsqueeze(0).to(device)
    X_partial = X_partial.unsqueeze(0).to(device)
    completions = torch.stack(completions, dim=0).unsqueeze(0).to(device)  # [1, N, 3, H, W]
    
    # [1, N, 3, H, W] → [1, N*3, H, W]로 변환
    B, N, C, H, W = completions.size()
    completions_flat = completions.view(B, N*C, H, W)
    
    # 입력 준비
    inputs = torch.cat([X_partial, M, completions_flat], dim=1)
    
    # Transformer 순방향 전파
    with torch.no_grad():
        weight_maps = model(inputs)
    
    # 가중치를 이용해 완성 이미지들 융합
    weight_maps_expanded = weight_maps.unsqueeze(2)  # [1, N, 1, H, W]
    fused = (weight_maps_expanded * completions).sum(dim=1)  # [1, 3, H, W]
    
    # 융합된 이미지를 마스크 영역에만 적용하여 최종 입력 이미지 생성
    X_input = X_partial.clone()
    mask_expanded = M.expand_as(X_partial)
    X_input = X_input * (1 - mask_expanded) + fused * mask_expanded
    
    # 결과 시각화 및 저장
    os.makedirs(output_dir, exist_ok=True)
    
    # 시각화 이미지 변환 ([-1, 1] → [0, 1])
    def convert_tensor(t):
        return (t[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2.0
    
    X_partial_vis = convert_tensor(X_partial)
    X_vis = convert_tensor(X)
    M_vis = M[0].detach().cpu().numpy().repeat(3, axis=0).transpose(1, 2, 0)
    fused_vis = convert_tensor(fused)
    X_input_vis = convert_tensor(X_input)
    
    # 완성 이미지들 변환
    comp_vis = []
    for i in range(N):
        comp = convert_tensor(completions[:, i])
        comp_vis.append(comp)
    
    # 가중치 맵 변환
    weight_maps_vis = []
    for i in range(N):
        w_map = weight_maps[0, i].detach().cpu().numpy()
        w_map = np.stack([w_map, w_map, w_map], axis=2)  # [H, W, 3]으로 변환
        weight_maps_vis.append(w_map)
    
    # 이미지 그리드 생성
    fig, axs = plt.subplots(3, N + 3, figsize=(3*(N + 3), 9))
    
    # 첫 번째 행: 부분 이미지, 마스크, 원본 이미지, 완성 이미지들
    axs[0, 0].imshow(X_partial_vis)
    axs[0, 0].set_title('부분 이미지')
    axs[0, 1].imshow(M_vis, cmap='gray')
    axs[0, 1].set_title('마스크')
    axs[0, 2].imshow(X_vis)
    axs[0, 2].set_title('원본 이미지')
    
    for i in range(N):
        axs[0, i+3].imshow(comp_vis[i])
        axs[0, i+3].set_title(f'완성 {i+1}')
    
    # 두 번째 행: 가중치 맵들
    for i in range(3):
        axs[1, i].axis('off')
    
    for i in range(N):
        axs[1, i+3].imshow(weight_maps_vis[i])
        axs[1, i+3].set_title(f'가중치 맵 {i+1}')
    
    # 세 번째 행: 융합된 이미지, 최종 입력 이미지, 원본 이미지 비교
    axs[2, 0].imshow(fused_vis)
    axs[2, 0].set_title('융합된 이미지')
    axs[2, 1].imshow(X_input_vis)
    axs[2, 1].set_title('최종 입력')
    axs[2, 2].imshow(X_vis)
    axs[2, 2].set_title('원본 (타겟)')
    
    for i in range(N):
        axs[2, i+3].axis('off')
    
    # 모든 축의 축 제거
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 시각화 저장
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{image_id}_visualization.png")
    plt.close()
    
    # 최종 입력 이미지 저장 ([-1, 1] → [0, 255])
    X_input_img = (X_input[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2.0 * 255
    X_input_img = X_input_img.astype(np.uint8)
    Image.fromarray(X_input_img).save(f"{output_dir}/{image_id}_fused.png")
    
    # 개별 이미지 저장
    # 마스크만 적용된 개별 완성 이미지 저장
    for i in range(N):
        if not os.path.exists(os.path.join(data_dir, f"{image_id}_comp{i+1}.png")):
            continue
        
        # 개별 완성 이미지에 마스크 적용
        single_comp = completions[0, i]
        single_input = X_partial.clone()
        single_input = single_input * (1 - mask_expanded) + single_comp * mask_expanded
        
        # 이미지 저장 ([-1, 1] → [0, 255])
        single_img = (single_input[0].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2.0 * 255
        single_img = single_img.astype(np.uint8)
        Image.fromarray(single_img).save(f"{output_dir}/{image_id}_comp{i+1}_masked.png")
    
    # 각 가중치 맵 개별 저장
    for i in range(N):
        if not os.path.exists(os.path.join(data_dir, f"{image_id}_comp{i+1}.png")):
            continue
        
        # 가중치 맵을 히트맵으로 저장
        w_map = weight_maps[0, i].detach().cpu().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(w_map, cmap='viridis')
        plt.colorbar()
        plt.title(f'가중치 맵 {i+1}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{image_id}_weight{i+1}.png")
        plt.close()
    
    return X_input.detach().cpu()

def run_inference(args):
    """
    학습된 Transformer 모델을 이용한 추론 실행
    
    Args:
        args: 인수 객체
    """
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")
    
    # 모델 로드
    n_channels = 3 + 1 + 3 * args.n_max_completions
    model = AmodalFusionTransformer(
        in_channels=n_channels, 
        n_completions=args.n_max_completions,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)
    
    # 체크포인트 로드
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # DDP 모델의 경우 'module.' 접두사 처리
    if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            new_state_dict[k.replace('module.', '')] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"체크포인트 로드됨: {args.checkpoint}, 에폭 {checkpoint.get('epoch', 'N/A')}")
    
    # 평가 모드 설정
    model.eval()
    
    # 이미지 ID 목록 로드
    if args.image_list:
        with open(args.image_list, 'r') as f:
            image_ids = [line.strip() for line in f]
    else:
        # 디렉토리에서 모든 partial 이미지 찾기
        image_ids = []
        for file in os.listdir(args.data_dir):
            if file.endswith('_partial.jpg'):
                image_id = file[:-12]  # '_partial.jpg' 제거
                image_ids.append(image_id)
    
    print(f"총 {len(image_ids)}개 이미지 처리 중...")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 각 이미지 처리
    for image_id in tqdm(image_ids):
        process_image(
            image_id,
            args.data_dir,
            model,
            args.img_size,
            args.n_max_completions,
            device,
            args.output_dir
        )
        
    print(f"모든 이미지 처리 완료. 결과는 {args.output_dir}에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description='Transformer 기반 AmodalFusionNet 추론을 위한 스크립트')
    
    # 데이터 관련 인수
    parser.add_argument('--data_dir', type=str, required=True, help='데이터셋 디렉토리')
    parser.add_argument('--image_list', type=str, help='처리할 이미지 ID 목록 파일 (없으면 디렉토리에서 자동 탐색)')
    parser.add_argument('--img_size', type=int, default=256, help='이미지 크기')
    parser.add_argument('--n_max_completions', type=int, default=5, help='최대 완성 이미지 수')
    
    # 모델 관련 인수
    parser.add_argument('--checkpoint', type=str, required=True, help='로드할 체크포인트 파일 경로')
    parser.add_argument('--d_model', type=int, default=256, help='Transformer 모델 차원')
    parser.add_argument('--nhead', type=int, default=8, help='멀티헤드 어텐션의 헤드 수')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='인코더 레이어 수')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='디코더 레이어 수')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='피드포워드 네트워크의 히든 차원')
    parser.add_argument('--dropout', type=float, default=0.1, help='드롭아웃 비율')
    
    # 출력 관련 인수
    parser.add_argument('--output_dir', type=str, default='outputs', help='결과 저장 디렉토리')
    
    # 설정 파일을 통한 인수 지정
    parser.add_argument('--config', type=str, help='YAML 설정 파일')
    
    args = parser.parse_args()
    
    # 설정 파일을 통한 인수 로드 (제공된 경우)
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if not parser.get_default(key):  # 파서에 정의된 인수만 설정
                    continue
                setattr(args, key, value)
    
    # Transformer 추론 실행
    run_inference(args)

if __name__ == '__main__':
    main()