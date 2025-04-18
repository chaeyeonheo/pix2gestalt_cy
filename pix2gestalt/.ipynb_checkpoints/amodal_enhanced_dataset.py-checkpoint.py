import numpy as np
import torch
import cv2
import os
import sys
from PIL import Image
from tqdm import tqdm
import gc
from omegaconf import OmegaConf

# inference.py에서 필요한 함수들 가져오기
from inference import load_model_from_config, get_sam_predictor, run_sam, run_inference

def create_amodal_enhanced_dataset(
    input_dir,           # 마스킹된 이미지가 있는 디렉토리
    output_dir,          # 향상된 이미지를 저장할 디렉토리
    model_config_path=None,   # pix2gestalt 모델 설정 경로
    model_ckpt_path=None,     # pix2gestalt 모델 체크포인트 경로
    sam_model_type='vit_h', # SAM 모델 타입
    device='cuda',       # 추론에 사용할 장치
    guidance_scale=2.0,  # 확산 가이던스 스케일
    ddim_steps=50,       # 확산 스텝 수
    batch_size=1         # 처리를 위한 배치 크기
):
    # 기본 경로 설정 (제공되지 않은 경우)
    if model_config_path is None:
        model_config_path = 'config.yaml'  # 기본 설정 파일 경로
    if model_ckpt_path is None:
        model_ckpt_path = './ckpt/epoch=000005.ckpt'  # 기본 체크포인트 경로
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # pix2gestalt 모델 로드
    config = OmegaConf.load(model_config_path)
    model = load_model_from_config(config, model_ckpt_path, device)
    
    # SAM 예측기 가져오기
    sam_predictor = get_sam_predictor(model_type=sam_model_type, device=device)

    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in tqdm(image_files, desc="이미지 처리 중"):
        img_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        
        # 출력이 이미 존재하면 건너뜀
        if os.path.exists(output_path):
            print(f"{img_file} 파일은 이미 처리되었으므로 건너뜁니다")
            continue
        
        # 이미지 로드
        input_image = cv2.imread(img_path)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # 마스크 생성 (흰색/검은색 영역이 마스킹됨을 가정)
        # 흰색 마스크 (255, 255, 255)
        white_mask = np.all(input_image == 255, axis=2)
        # 검은색 마스크 (0, 0, 0)
        black_mask = np.all(input_image == 0, axis=2)
        # 결합된 마스크
        mask = white_mask | black_mask
        
        # 이진 마스크로 변환 (마스킹된 영역은 1, 가시적인 영역은 0)
        binary_mask = mask.astype(np.uint8) * 255
        
        # SAM 이미지 설정
        sam_predictor.set_image(input_image)
        
        # 가시적인 영역에서 객체 감지
        # 그리드 포인트를 사용하여 SAM에 프롬프트 제공
        h, w = input_image.shape[:2]
        grid_size = 32  # 포인트 프롬프트의 그리드 크기
        
        # 결과 이미지 (입력 이미지로 초기화)
        result_image = input_image.copy()
        
        # 채워진 마스크 픽셀 추적
        filled_mask = np.zeros_like(binary_mask)
        
        # SAM용 그리드 포인트 생성
        points = []
        for y in range(grid_size, h, grid_size):
            for x in range(grid_size, w, grid_size):
                # 가시적인 영역의 포인트만 사용
                if not mask[y, x]:
                    points.append(([x, y], 1))  # 전경 포인트
        
        print(f"SAM 프롬프트용 {len(points)}개 포인트 생성됨")
        
        # 메모리 문제를 피하기 위해 작은 배치로 처리
        batch_points = [points[i:i+10] for i in range(0, len(points), 10)]
        
        for point_batch in batch_points:
            if not point_batch:
                continue
                
            try:
                # 이 배치의 포인트에 대한 가시적 마스크 가져오기
                visible_mask, _ = run_sam(sam_predictor, point_batch)
                
                # 유효한 마스크가 생성되지 않으면 건너뜀
                if visible_mask is None or visible_mask.max() == 0:
                    continue
                
                # pix2gestalt를 실행하여 객체 완성
                completions = run_inference(
                    input_image,
                    visible_mask,
                    model,
                    guidance_scale,
                    n_samples=1,  # 샘플 하나만 생성
                    ddim_steps=ddim_steps,
                    device=device
                )
                
                if completions and len(completions) > 0:
                    completion = completions[0]
                    
                    # 마스크된 영역만 업데이트
                    # visible_mask가 1이고 binary_mask가 1이고 filled_mask가 0인 마스크 생성
                    update_mask = (visible_mask > 0) & (binary_mask > 0) & (filled_mask == 0)
                    update_mask = update_mask.astype(bool)
                    
                    # 마스크된 영역 업데이트
                    if update_mask.any():
                        # 업데이트 마스크 영역에만 완성된 이미지 적용
                        for c in range(3):  # RGB 채널
                            result_image[:,:,c] = np.where(update_mask, completion[:,:,c], result_image[:,:,c])
                        
                        # 채워진 마스크 업데이트
                        filled_mask = filled_mask | update_mask.astype(np.uint8)
                
            except Exception as e:
                print(f"배치 처리 중 오류 발생: {e}")
                continue
            
            # 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 결과 이미지 저장
        output_img = Image.fromarray(result_image)
        output_img.save(output_path)
        
        fill_ratio = filled_mask.sum() / (binary_mask.sum() + 1e-8) * 100  # 0으로 나누기 방지
        print(f"{img_file} 처리 완료. 마스크 영역의 {fill_ratio:.2f}% 채워짐")
    
    # 최종 정리
    del model, sam_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 메인 함수
def main():
    import argparse
    parser = argparse.ArgumentParser(description='아모달 세그멘테이션을 활용한 인페인팅 데이터셋 생성')
    parser.add_argument('--input_dir', type=str, required=True, help='마스킹된 이미지가 있는 디렉토리')
    parser.add_argument('--output_dir', type=str, required=True, help='향상된 이미지를 저장할 디렉토리')
    parser.add_argument('--model_config', type=str, default='config.yaml', help='pix2gestalt 모델 설정 경로')
    parser.add_argument('--model_ckpt', type=str, default='./ckpt/epoch=000005.ckpt', help='pix2gestalt 모델 체크포인트 경로')
    parser.add_argument('--sam_model', type=str, default='vit_h', help='SAM 모델 타입 (vit_b, vit_l, vit_h)')
    parser.add_argument('--device', type=str, default='cuda', help='사용할 장치 (cuda 또는 cpu)')
    parser.add_argument('--guidance_scale', type=float, default=2.0, help='확산 가이던스 스케일')
    parser.add_argument('--ddim_steps', type=int, default=50, help='확산 스텝 수')
    args = parser.parse_args()
    
    create_amodal_enhanced_dataset(
        args.input_dir,
        args.output_dir,
        args.model_config,
        args.model_ckpt,
        args.sam_model,
        args.device,
        args.guidance_scale,
        args.ddim_steps
    )

if __name__ == "__main__":
    main()