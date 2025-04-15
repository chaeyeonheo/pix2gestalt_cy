import os
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
from omegaconf import OmegaConf

# inference.py에서 필요한 함수들 임포트
from inference import (
    load_model_from_config,
    run_inference
)

def process_with_external_masks(
    image_dir,
    mask_dir,
    output_dir,
    config_path,
    ckpt_path,
    gpu_ids=None,
    guidance_scale=2.0,
    n_samples=4,
    ddim_steps=200
):
    """외부 마스크 파일을 사용하여 이미지 처리"""
    # GPU 설정
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [int(id) for id in gpu_ids.split(',')]
    
    print(f"Using GPUs: {gpu_ids}")
    
    # 이미지 및 마스크 파일 찾기
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # GPU마다 모델 로드
    models = {}
    for gpu_id in gpu_ids:
        device = f"cuda:{gpu_id}"
        print(f"Loading model on {device}")
        config = OmegaConf.load(config_path)
        model = load_model_from_config(config, ckpt_path, device)
        models[gpu_id] = model
    
    # 이미지 처리
    for i, image_path in enumerate(image_paths):
        # 대응하는 마스크 파일 찾기 (같은 이름이지만 다른 디렉토리)
        mask_path = mask_dir / f"{image_path.stem}_mask.png"
        if not mask_path.exists():
            print(f"Mask not found for {image_path}, skipping")
            continue
        
        # 간단한 로드 밸런싱: 이미지 인덱스에 따라 GPU 할당
        gpu_id = gpu_ids[i % len(gpu_ids)]
        device = f"cuda:{gpu_id}"
        model = models[gpu_id]
        
        # 이미지 및 마스크 읽기
        print(f"Processing {image_path} with mask {mask_path}")
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            print(f"Error reading image or mask, skipping")
            continue
        
        # 크기 조정
        image_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # 이진 마스크로 변환 (필요한 경우)
        _, mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
        
        # pix2gestalt 실행
        pred_reconstructions = run_inference(
            image_resized, 
            mask_binary, 
            model, 
            guidance_scale, 
            n_samples, 
            ddim_steps,
            device=device
        )
        
        # 결과 저장
        img_output_dir = os.path.join(output_dir, image_path.stem)
        os.makedirs(img_output_dir, exist_ok=True)
        
        # 원본 이미지 저장
        cv2.imwrite(os.path.join(img_output_dir, f"{image_path.stem}_original.png"), image_resized)
        
        # 마스크 저장
        cv2.imwrite(os.path.join(img_output_dir, f"{image_path.stem}_mask.png"), mask_binary)
        
        # 결과 이미지 저장
        for j, pred_image in enumerate(pred_reconstructions):
            result_path = os.path.join(img_output_dir, f"{image_path.stem}_result_{j}.png")
            cv2.imwrite(result_path, pred_image)
        
        # 메모리 정리
        torch.cuda.empty_cache()
        
        print(f"Results saved to {img_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Process images with external masks using pix2gestalt")
    parser.add_argument("--image_dir", required=True, help="Directory containing input images")
    parser.add_argument("--mask_dir", required=True, help="Directory containing mask images")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--config", default="./configs/sd-finetune-pix2gestalt-c_concat-256.yaml", help="Config path")
    parser.add_argument("--ckpt", default="./ckpt/epoch=000005.ckpt", help="Checkpoint path")
    parser.add_argument("--gpu_ids", default=None, help="Comma-separated GPU IDs to use (e.g., '0,1,2')")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="Diffusion guidance scale")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--ddim_steps", type=int, default=200, help="Number of DDIM steps")
    
    args = parser.parse_args()
    
    process_with_external_masks(
        args.image_dir,
        args.mask_dir,
        args.output_dir,
        args.config,
        args.ckpt,
        args.gpu_ids,
        args.guidance_scale,
        args.n_samples,
        args.ddim_steps
    )

if __name__ == "__main__":
    main()