import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path

# amodal_fusion_dataset.py의 AmodalFusionDataset 클래스 수정

class AmodalFusionDataset(Dataset):
    def __init__(self, data_dir, n_max_completions=5, transform=None):
        """
        이미지별 폴더 구조를 사용하는 Amodal Fusion 데이터셋 초기화
        
        Args:
            data_dir: 데이터셋 디렉토리 경로
            n_max_completions: 최대 completion 이미지 수
            transform: 이미지 변환 함수
        """
        self.data_dir = Path(data_dir)
        self.n_max_completions = n_max_completions
        self.transform = transform
        self.ids = []
        
        # 폴더에서 유효한 이미지 ID 찾기
        for item in os.listdir(self.data_dir):
            item_path = self.data_dir / item
            
            # 디렉토리인 경우만 처리
            if os.path.isdir(item_path):
                image_id = item  # 폴더 이름이 이미지 ID
                
                # 필요한 모든 파일이 존재하는지 확인
                required_files = [
                    f"{image_id}.jpg",
                    f"{image_id}_mask.png",
                    f"{image_id}_partial.jpg"
                ]
                
                # 최소한 하나의 completion 파일이 있는지 확인
                has_comp = False
                for i in range(1, self.n_max_completions + 1):
                    if os.path.exists(item_path / f"{image_id}_comp{i}.png"):
                        has_comp = True
                        break
                
                # 모든 필수 파일이 존재하고 최소 하나의 completion 파일이 있으면 ID 추가
                if all(os.path.exists(item_path / file) for file in required_files) and has_comp:
                    self.ids.append(image_id)
        
        print(f"{self.data_dir} 폴더에서 {len(self.ids)}개의 유효한 이미지 ID 발견됨")
    
    def __getitem__(self, idx):
        """데이터셋에서 하나의 샘플 가져오기"""
        img_id = self.ids[idx]
        item_path = self.data_dir / img_id  # 이미지 ID에 해당하는 폴더 경로
        
        # 원본 이미지, 마스크, 부분 이미지 로드
        X = Image.open(item_path / f"{img_id}.jpg").convert("RGB")
        M = Image.open(item_path / f"{img_id}_mask.png").convert("L")
        X_partial = Image.open(item_path / f"{img_id}_partial.jpg").convert("RGB")
        
        # 모든 completion 이미지 로드
        completions = []
        for j in range(1, self.n_max_completions + 1):
            comp_path = item_path / f"{img_id}_comp{j}.png"
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

def create_dataloaders(train_dir, val_dir, batch_size=8, img_size=256, n_max_completions=5, num_workers=4):
    """
    훈련 및 검증 데이터로더 생성
    
    Args:
        train_dir: 훈련 데이터 디렉토리
        val_dir: 검증 데이터 디렉토리
        batch_size: 배치 크기
        img_size: 이미지 크기
        n_max_completions: 최대 completion 이미지 수
        num_workers: 데이터 로더 워커 수
        
    Returns:
        train_loader, val_loader: 훈련 및 검증 데이터로더
    """
    # 이미지 변환 설정
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 데이터셋 생성
    train_dataset = AmodalFusionDataset(
        train_dir,
        n_max_completions=n_max_completions,
        transform=transform
    )
    
    val_dataset = AmodalFusionDataset(
        val_dir,
        n_max_completions=n_max_completions,
        transform=transform
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def create_distributed_dataloaders(train_dir, val_dir, world_size, rank, batch_size=8, 
                                 img_size=256, n_max_completions=5, num_workers=4):
    """
    분산 학습을 위한 데이터로더 생성
    
    Args:
        train_dir: 훈련 데이터 디렉토리
        val_dir: 검증 데이터 디렉토리
        world_size: 총 프로세스 수
        rank: 현재 프로세스 순위
        batch_size: 배치 크기
        img_size: 이미지 크기
        n_max_completions: 최대 completion 이미지 수
        num_workers: 데이터 로더 워커 수
        
    Returns:
        train_loader, val_loader: 분산 훈련 및 검증 데이터로더, train_sampler
    """
    # 이미지 변환 설정
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 데이터셋 생성
    train_dataset = AmodalFusionDataset(
        train_dir,
        n_max_completions=n_max_completions,
        transform=transform
    )
    
    val_dataset = AmodalFusionDataset(
        val_dir,
        n_max_completions=n_max_completions,
        transform=transform
    )
    
    # 분산 샘플러 사용
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # 분산 샘플러에서 셔플 처리
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler