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
    get_sam_predictor,
    run_inference,
    process_input
)

def process_image(
    image_path, 
    output_dir, 
    model, 
    device, 
    points=None, 
    guidance_scale=2.0, 
    n_samples=4, 
    ddim_steps=200
):
    """단일 이미지 처리 함수"""
    # 이미지 읽기 및 전처리
    print(f"Processing image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    # 이미지 크기 조정 (256x256)
    image_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    
    # SAM 예측기 설정
    predictor = get_sam_predictor(device=device, model_type='vit_h')
    predictor.set_image(image_resized)
    
    # 포인트가 제공되지 않은 경우 이미지 중앙점 사용
    if points is None:
        h, w = image_resized.shape[:2]
        points = [([w//2, h//2], 1)]  # 이미지 중앙에 양성 포인트 하나
    
    # SAM으로 마스크 생성
    input_points = [p for p, _ in points]
    input_labels = [int(l) for _, l in points]
    
    masks, _, _ = predictor.predict(
        point_coords=np.array(input_points),
        point_labels=input_labels,
        multimask_output=False,
    )
    
    visible_mask = 255 * np.squeeze(masks).astype(np.uint8)  # (256, 256)
    
    # pix2gestalt 실행
    pred_reconstructions = run_inference(
        image_resized, 
        visible_mask, 
        model, 
        guidance_scale, 
        n_samples, 
        ddim_steps,
        device=device
    )
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 원본 이미지 저장
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.png"), image_resized)
    
    # 마스크 저장
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, visible_mask)
    
    # 결과 이미지 저장
    for i, pred_image in enumerate(pred_reconstructions):
        result_path = os.path.join(output_dir, f"{base_name}_result_{i}.png")
        cv2.imwrite(result_path, pred_image)
    
    # 원본 + 마스크 오버레이 저장
    overlay = image_resized.copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[visible_mask > 0] = [0, 255, 0]  # 녹색으로 마스크 표시
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    
    print(f"Results saved to {output_dir}")
    return pred_reconstructions, visible_mask

def process_batch(
    input_dir,
    output_dir,
    config_path,
    ckpt_path,
    gpu_ids=None,
    guidance_scale=2.0,
    n_samples=4,
    ddim_steps=200
):
    """배치 처리 함수"""
    # GPU 설정
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [int(id) for id in gpu_ids.split(',')]
    
    print(f"Using GPUs: {gpu_ids}")
    
    # 이미지 파일 찾기
    input_dir = Path(input_dir)
    image_paths = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
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
        # 간단한 로드 밸런싱: 이미지 인덱스에 따라 GPU 할당
        gpu_id = gpu_ids[i % len(gpu_ids)]
        device = f"cuda:{gpu_id}"
        model = models[gpu_id]
        
        # 이미지별 출력 디렉토리
        img_output_dir = os.path.join(output_dir, image_path.stem)
        
        # 이미지 처리
        process_image(
            image_path, 
            img_output_dir, 
            model, 
            device, 
            guidance_scale=guidance_scale, 
            n_samples=n_samples, 
            ddim_steps=ddim_steps
        )
        
        # 메모리 정리
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Batch process images with pix2gestalt")
    parser.add_argument("--input_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--config", default="./configs/sd-finetune-pix2gestalt-c_concat-256.yaml", help="Config path")
    parser.add_argument("--ckpt", default="./ckpt/epoch=000005.ckpt", help="Checkpoint path")
    parser.add_argument("--gpu_ids", default=None, help="Comma-separated GPU IDs to use (e.g., '0,1,2')")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="Diffusion guidance scale")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--ddim_steps", type=int, default=200, help="Number of DDIM steps")
    
    args = parser.parse_args()
    
    process_batch(
        args.input_dir,
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