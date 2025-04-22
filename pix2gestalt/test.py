import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import sys
import os
import cv2
import argparse

# 필요한 모듈 임포트
from inference import load_model_from_config, get_sam_predictor, run_sam, run_inference

def main():
    parser = argparse.ArgumentParser(description="Run pix2gestalt model with mask")
    parser.add_argument("--mode", type=str, default="sam_auto", choices=["existing_mask", "sam_auto"],
                        help="Mode to generate mask: 'existing_mask' or 'sam_auto'")
    parser.add_argument("--mask_path", type=str, default=None, 
                        help="Path to existing mask image (only used in 'existing_mask' mode)")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    args = parser.parse_args()

    # 설정 파일 로드
    config_path = "./configs/sd-finetune-pix2gestalt-c_concat-256.yaml"
    config = OmegaConf.load(config_path)

    # 모델 로드
    checkpoint_path = "./ckpt/epoch=000005.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_config(config, checkpoint_path, device)

    # 입력 이미지 로드
    input_image = np.array(Image.open(args.input_path).resize((256, 256)))
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 마스크 생성 방법에 따라 처리
    if args.mode == "existing_mask":
        # 1. 기존 마스크 사용
        if args.mask_path is None:
            raise ValueError("Mask path must be provided in 'existing_mask' mode")
        
        # 마스크 로드 및 전처리
        mask = np.array(Image.open(args.mask_path).resize((256, 256)))
        if len(mask.shape) == 3:  # 채널이 있는 경우
            if mask.shape[2] == 3:  # RGB 마스크
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # 이진화 (확실히 0 또는 255 값만 가지도록)
        _, visible_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
    else:  # "sam_auto" 모드
        # 2. SAM 자동 추출 사용
        predictor = get_sam_predictor(model_type='vit_h', device=device, image=input_image)
        
        # SAM의 자동 마스크 생성 기능 사용
        masks = predictor.model.generate().masks
        
        # 가장 큰 마스크 선택 (일반적으로 주요 객체)
        if len(masks) > 0:
            # 각 마스크의 크기(픽셀 수) 계산
            mask_sizes = [np.sum(mask) for mask in masks]
            # 가장 큰 마스크 선택
            largest_mask_idx = np.argmax(mask_sizes)
            visible_mask = masks[largest_mask_idx].astype(np.uint8) * 255
        else:
            # 대안: 중앙 객체 선택
            print("SAM automatic mask generation failed, falling back to center point prompt")
            selected_points = [([input_image.shape[1]//2, input_image.shape[0]//2], 1)]
            visible_mask, _ = run_sam(predictor, selected_points)
    
    # 원본 이미지와 마스크 저장
    Image.fromarray(input_image).save(os.path.join(args.output_dir, "input.png"))
    Image.fromarray(visible_mask).save(os.path.join(args.output_dir, "mask.png"))
    
    # 모델 추론 실행
    results = run_inference(
        input_image=input_image,
        visible_mask=visible_mask,
        model=model,
        guidance_scale=2.0,
        n_samples=4,  # 여러 샘플 생성
        ddim_steps=50,
        device=device
    )

    # 결과 저장
    for i, result in enumerate(results):
        Image.fromarray(result).save(os.path.join(args.output_dir, f"result_{i}.png"))
    
    print(f"Processing complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()