import os
import argparse
import glob
import cv2
import numpy as np
import torch
from PIL import Image
import random
from tqdm import tqdm

# 마스크 유틸리티 함수들 임포트
from mask_utils import (
    SeededRandomMask, SeededRandomBrush, ReplicatedRandomMask, 
    BatchSeededRandomMask, grid_probability_mask, uniform_grid_probability_mask,
    box_grid_probability_mask, random_box_grid_probability_mask, 
    grid_mask, concentric_box_mask, write_images
)

def apply_mask_to_image(image, mask, color=(255, 255, 255)):
    """
    이미지에 마스크 적용 (마스크 영역을 지정된 색상으로 채움)
    
    Args:
        image: 원본 이미지 (H, W, C)
        mask: 마스크 (H, W) 또는 (1, H, W), 0은 마스킹할 영역, 1은 보존할 영역
        color: 마스킹 영역을 채울 색상 (기본값: 흰색)
    
    Returns:
        마스킹된 이미지
    """
    # 마스크 형태 확인 및 조정
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = mask[0]  # (1, H, W) -> (H, W)
    
    # 마스크 반전 (0: 마스킹, 1: 보존)
    binary_mask = (1 - mask).astype(np.bool)
    
    # 마스킹된 이미지 생성
    masked_image = image.copy()
    
    # 이미지 채널 수에 따라 처리
    if len(image.shape) == 2:  # 그레이스케일
        masked_image[binary_mask] = color[0]
    else:  # RGB 또는 RGBA
        for c in range(min(len(color), image.shape[2])):
            masked_image[binary_mask, c] = color[c]
            
    return masked_image

def create_masks_for_image(image_path, output_dir, mask_policies, mask_ratio=0.5, seed=42, num_samples=1):
    """
    이미지에 여러 마스크 정책 적용 및 결과 저장
    
    Args:
        image_path: 입력 이미지 경로
        output_dir: 출력 디렉토리 기본 경로
        mask_policies: 사용할 마스크 정책 목록
        mask_ratio: 마스크 비율 (0-1)
        seed: 랜덤 시드
        num_samples: 각 정책당 생성할 샘플 수
    """
    # 이미지 로드 및 전처리
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return
    
    # BGR -> RGB 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지 크기 정의 (모든 이미지 256x256으로 리사이즈)
    target_size = 256
    image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # 이미지 파일명 추출 (확장자 제외)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 각 마스크 정책에 대해 처리
    for policy in mask_policies:
        # 정책별 출력 디렉토리 생성
        policy_dir = os.path.join(output_dir, policy)
        os.makedirs(policy_dir, exist_ok=True)
        
        # 마스크 생성을 위한 텐서 shape 정의
        shape = (1, 3, target_size, target_size)  # 배치 크기 1, RGB 채널
        device = "cpu"  # CPU에서 실행
        
        # 각 샘플 생성
        for sample_idx in range(num_samples):
            current_seed = seed + sample_idx  # 샘플마다 다른 시드 사용
            
            # 정책에 따른 마스크 생성
            if policy == 'seeded_random':
                hole_range = [max(0.01, mask_ratio * 0.9), min(0.99, mask_ratio * 1.1)]
                mask_np = SeededRandomMask(target_size, hole_range=hole_range, seed=current_seed)
                mask = mask_np[0]  # (1, H, W) -> (H, W)
                
            elif policy == 'seeded_brush':
                brush_intensity = 9
                mask = 1 - SeededRandomBrush(brush_intensity, target_size, seed=current_seed)
                
            elif policy == 'grid':
                grid_size = 16
                mask_tensor = grid_mask(shape, device, grid_size=grid_size, p=mask_ratio)
                mask = mask_tensor[0, 0].numpy()  # (B, C, H, W) -> (H, W)
                
            elif policy == 'concentric':
                num_boxes = 3
                box_gap = 20
                box_width = 10
                mask_tensor = concentric_box_mask(shape, device, num_boxes=num_boxes, 
                                               box_gap=box_gap, box_width=box_width)
                mask = mask_tensor[0, 0].numpy()
                
            elif policy == 'grid_probability':
                grid_size = 16
                p_range = (mask_ratio*0.5, mask_ratio*1.5)
                mask_tensor = grid_probability_mask(shape, device, grid_size=grid_size, p_range=p_range)
                mask = mask_tensor[0, 0].numpy()
                
            elif policy == 'uniform_grid':
                grid_size = 16
                mask_tensor = uniform_grid_probability_mask(shape, device, grid_size=grid_size, p=mask_ratio)
                mask = mask_tensor[0, 0].numpy()
                
            elif policy == 'box_grid':
                grid_size = 16
                mask_tensor = box_grid_probability_mask(shape, device, grid_size=grid_size, p=mask_ratio)
                mask = mask_tensor[0, 0].numpy()
                
            elif policy == 'random_box_grid':
                grid_size = 16
                p_range = (mask_ratio*0.5, mask_ratio*1.5)
                mask_tensor, _ = random_box_grid_probability_mask(shape, device, grid_size=grid_size, p_range=p_range)
                mask = mask_tensor[0, 0].numpy()
                
            else:
                print(f"알 수 없는 마스크 정책: {policy}, 기본 seeded_random 사용")
                hole_range = [max(0.01, mask_ratio * 0.9), min(0.99, mask_ratio * 1.1)]
                mask_np = SeededRandomMask(target_size, hole_range=hole_range, seed=current_seed)
                mask = mask_np[0]
            
            # 마스크 적용
            masked_image = apply_mask_to_image(image_resized, mask)
            
            # 결과 저장
            if num_samples > 1:
                output_path = os.path.join(policy_dir, f"{image_name}_sample{sample_idx}.png")
                mask_path = os.path.join(policy_dir, f"{image_name}_sample{sample_idx}_mask.png")
            else:
                output_path = os.path.join(policy_dir, f"{image_name}.png")
                mask_path = os.path.join(policy_dir, f"{image_name}_mask.png")
            
            # RGB -> BGR 변환 (OpenCV용)
            masked_image_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, masked_image_bgr)
            
            # 마스크도 저장
            cv2.imwrite(mask_path, mask * 255)  # 0-1 -> 0-255
            
    return image_name

def main():
    parser = argparse.ArgumentParser(description="이미지에 다양한 마스크 정책 적용 후 저장")
    parser.add_argument("--input_dir", required=True, help="입력 이미지 디렉토리")
    parser.add_argument("--output_dir", required=True, help="출력 디렉토리")
    parser.add_argument("--mask_policies", nargs='+', default=['seeded_random', 'seeded_brush', 'grid', 'concentric'],
                        help="사용할 마스크 정책 목록")
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="마스크 비율 (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--num_images", type=int, default=None, 
                        help="처리할 이미지 수 (기본값: 모든 이미지)")
    parser.add_argument("--num_samples", type=int, default=1, 
                        help="각 정책당 생성할 샘플 수")
    
    args = parser.parse_args()
    
    # 입력 디렉토리에서 이미지 파일 찾기
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png']
    
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(args.input_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(args.input_dir, f"*{ext.upper()}")))
    
    # 이미지 파일 수 제한 (설정된 경우)
    if args.num_images is not None and args.num_images > 0:
        # 랜덤 선택 대신 앞에서부터 선택
        image_files = image_files[:args.num_images]
    
    print(f"처리할 이미지 수: {len(image_files)}")
    print(f"사용할 마스크 정책: {args.mask_policies}")
    
    # 각 이미지에 마스크 적용
    for image_path in tqdm(image_files, desc="이미지 처리 중"):
        create_masks_for_image(
            image_path, 
            args.output_dir, 
            args.mask_policies, 
            args.mask_ratio, 
            args.seed,
            args.num_samples
        )
    
    print(f"모든 마스크 적용 완료. 결과는 {args.output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()