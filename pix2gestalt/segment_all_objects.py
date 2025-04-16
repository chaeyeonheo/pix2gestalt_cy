import os
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import multiprocessing
from functools import partial
import time
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# inference.py에서 필요한 함수들 임포트
from inference import (
    load_model_from_config,
    run_inference
)

class AmodalProcessor:
    def __init__(self, config_path, ckpt_path, device, sam_model_type='vit_h', verbose=False):
        """
        Amodal segmentation 처리를 위한 클래스 초기화
        
        Args:
            config_path: pix2gestalt 모델 설정 파일 경로
            ckpt_path: pix2gestalt 모델 체크포인트 경로
            device: 사용할 CUDA 디바이스 (예: 'cuda:0')
            sam_model_type: SAM 모델 타입 ('vit_h', 'vit_l', 'vit_b')
            verbose: 상세 로그 출력 여부
        """
        self.device = device
        self.verbose = verbose
        
        # pix2gestalt 모델 로드
        if verbose:
            print(f"Loading pix2gestalt model from {ckpt_path} on {device}")
        config = OmegaConf.load(config_path)
        self.model = load_model_from_config(config, ckpt_path, device)
        
        # SAM 모델 로드 및 자동 마스크 생성기 초기화
        if verbose:
            print(f"Loading SAM model ({sam_model_type}) on {device}")
        
        # SAM 모델 경로 설정
        sam_models = {
            'vit_b': './ckpt/sam_vit_b.pth',
            'vit_l': './ckpt/sam_vit_l.pth',
            'vit_h': './ckpt/sam_vit_h.pth'
        }
        
        # SAM 모델 로드
        sam = sam_model_registry[sam_model_type](checkpoint=sam_models[sam_model_type])
        sam.to(device)
        
        # 자동 마스크 생성기 설정
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,         # 더 많은 포인트로 세밀한 객체 감지
            pred_iou_thresh=0.88,       # IoU 점수 임계값
            stability_score_thresh=0.95, # 안정성 점수 임계값
            crop_n_layers=1,            # 크롭 레이어 수
            crop_n_points_downscale_factor=2,  # 크롭당 포인트 다운스케일 팩터
            min_mask_region_area=100    # 최소 마스크 영역 크기
        )
    
    def generate_masks(self, image):
        """
        이미지에서 모든 객체의 마스크 생성
        
        Args:
            image: 입력 이미지 (numpy array, RGB)
            
        Returns:
            생성된 마스크 목록 (각 마스크는 segment_anything의 형식)
        """
        if self.verbose:
            print("Generating object masks...")
        
        # SAM 자동 마스크 생성
        masks = self.mask_generator.generate(image)
        
        if self.verbose:
            print(f"Found {len(masks)} objects")
        
        return masks
    
    def process_image_with_all_objects(self, image_path, output_dir, combined_base_dir, guidance_scale=2.0, n_samples=1, ddim_steps=200):
        """
        이미지에서 모든 객체를 감지하고 각각에 amodal segmentation 적용
        
        Args:
            image_path: 처리할 이미지 경로
            output_dir: 결과 저장 디렉토리
            combined_base_dir: 통합 시각화 저장 기본 디렉토리
            guidance_scale: 확산 모델 가이드 강도
            n_samples: 생성할 샘플 수
            ddim_steps: DDIM 스텝 수
        """
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Cannot read image {image_path}")
            return
        
        # BGR -> RGB 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 이미지 크기 조정 (256x256)
        target_size = 256
        image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        # 모든 객체 마스크 생성
        masks = self.generate_masks(image_resized)
        
        # 출력 디렉토리 생성
        image_name = Path(image_path).stem
        image_output_dir = os.path.join(output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # 통합 시각화 디렉토리 생성
        combined_images_dir = os.path.join(combined_base_dir, "combined")
        combined_masks_dir = os.path.join(combined_base_dir, "combined_masks")
        os.makedirs(combined_images_dir, exist_ok=True)
        os.makedirs(combined_masks_dir, exist_ok=True)
        
        # 원본 이미지 저장
        cv2.imwrite(os.path.join(image_output_dir, f"{image_name}_original.png"), 
                   cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
        
        # 모든 객체 시각화
        all_objects_vis = image_resized.copy()
        for i, mask_data in enumerate(masks):
            # 랜덤 색상 생성
            color = np.random.randint(0, 255, size=3).tolist()
            # 마스크 오버레이
            mask_binary = mask_data['segmentation']
            all_objects_vis[mask_binary] = [
                all_objects_vis[mask_binary, 0] * 0.5 + color[0] * 0.5,
                all_objects_vis[mask_binary, 1] * 0.5 + color[1] * 0.5,
                all_objects_vis[mask_binary, 2] * 0.5 + color[2] * 0.5
            ]
        
        # 모든 객체 시각화 저장
        cv2.imwrite(os.path.join(image_output_dir, f"{image_name}_all_objects.png"), 
                   cv2.cvtColor(all_objects_vis, cv2.COLOR_RGB2BGR))
        
        # 각 객체에 대한 amodal completion 결과 저장용 리스트
        all_object_results = []
        
        # 각 객체에 대해 amodal segmentation 적용
        for i, mask_data in enumerate(masks):
            # 마스크 추출
            mask_binary = mask_data['segmentation']
            mask_uint8 = mask_binary.astype(np.uint8) * 255
            
            # 객체별 디렉토리 생성
            object_dir = os.path.join(image_output_dir, f"object_{i:03d}")
            os.makedirs(object_dir, exist_ok=True)
            
            # 마스크 저장
            cv2.imwrite(os.path.join(object_dir, f"mask.png"), mask_uint8)
            
            # 마스크된 이미지 생성 및 저장
            masked_image = image_resized.copy()
            masked_image[~mask_binary] = [255, 255, 255]  # 배경을 흰색으로
            cv2.imwrite(os.path.join(object_dir, f"masked_input.png"), 
                       cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
            
            # Amodal completion 실행
            try:
                pred_reconstructions = run_inference(
                    image_resized,
                    mask_uint8,
                    self.model,
                    guidance_scale,
                    n_samples,
                    ddim_steps,
                    device=self.device
                )
                
                # 결과 저장
                for j, pred_image in enumerate(pred_reconstructions):
                    result_path = os.path.join(object_dir, f"amodal_result_{j}.png")
                    cv2.imwrite(result_path, pred_image)
                    
                    # 첫 번째 결과는 통합 시각화에 사용
                    if j == 0:
                        # BGR -> RGB 변환
                        pred_image_rgb = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
                        all_object_results.append((mask_binary, pred_image_rgb))
                
                if self.verbose:
                    print(f"Processed object {i} for {image_name}")
                    
            except Exception as e:
                print(f"Error processing object {i} in {image_path}: {e}")
        
        # 모든 amodal completion 결과를 통합한 시각화 생성
        # 1. 각 샘플별로 합성
        for sample_idx in range(min(n_samples, 1)):  # 일단 첫 번째 샘플만 합성
            # 기본 캔버스는 원본 이미지
            combined_vis = image_resized.copy()
            combined_mask = np.zeros((target_size, target_size), dtype=bool)
            
            # 객체 결과를 순서대로 합성
            for mask_binary, pred_image in all_object_results:
                # 현재까지 마스크된 영역과 겹치지 않는 부분만 합성
                non_overlapping = mask_binary & ~combined_mask
                if np.any(non_overlapping):
                    combined_vis[non_overlapping] = pred_image[non_overlapping]
                    combined_mask |= mask_binary
            
            # 통합 시각화 저장 (별도 폴더에)
            combined_path = os.path.join(combined_images_dir, f"{image_name}.png")
            cv2.imwrite(combined_path, cv2.cvtColor(combined_vis, cv2.COLOR_RGB2BGR))
            
            # 마스크도 저장 (별도 폴더에)
            mask_path = os.path.join(combined_masks_dir, f"{image_name}.png")
            cv2.imwrite(mask_path, combined_mask.astype(np.uint8) * 255)
        
        return image_name, len(masks)

# 각 GPU 프로세스가 실행할 작업자 함수
def worker_process(params):
    """GPU 워커 프로세스"""
    gpu_id = params['gpu_id']
    image_paths = params['image_paths']
    output_dir = params['output_dir']
    combined_base_dir = params['combined_base_dir']
    config_path = params['config_path']
    ckpt_path = params['ckpt_path']
    guidance_scale = params['guidance_scale']
    n_samples = params['n_samples']
    ddim_steps = params['ddim_steps']
    
    device = f"cuda:{gpu_id}"
    processor = AmodalProcessor(config_path, ckpt_path, device, verbose=True)
    
    results = []
    for image_path in image_paths:
        try:
            image_name, num_objects = processor.process_image_with_all_objects(
                image_path, output_dir, combined_base_dir, guidance_scale, n_samples, ddim_steps
            )
            results.append((image_name, num_objects))
            # 메모리 정리
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"GPU {gpu_id} - Error processing {image_path}: {e}")
    
    return results

def process_batch_parallel(
    input_dir,
    output_dir,
    config_path,
    ckpt_path,
    gpu_ids=None,
    guidance_scale=2.0,
    n_samples=1,
    ddim_steps=200,
    num_images=None,
    combined_output_subdir="all_amodal"
):
    """병렬로 여러 GPU를 사용하여 배치 처리"""
    # GPU 설정
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [int(id) for id in gpu_ids.split(',')]
    
    print(f"Using GPUs: {gpu_ids}")
    
    # 이미지 파일 찾기
    input_dir = Path(input_dir)
    image_paths = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if num_images is not None and num_images > 0:
        image_paths = image_paths[:num_images]
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 통합 시각화를 위한 기본 디렉토리 생성
    combined_base_dir = os.path.join(output_dir, combined_output_subdir)
    os.makedirs(combined_base_dir, exist_ok=True)
    
    # 통합 시각화 하위 디렉토리 생성
    combined_images_dir = os.path.join(combined_base_dir, "combined")
    combined_masks_dir = os.path.join(combined_base_dir, "combined_masks")
    os.makedirs(combined_images_dir, exist_ok=True)
    os.makedirs(combined_masks_dir, exist_ok=True)
    
    # 이미지를 GPU 수에 맞게 분배
    images_per_gpu = {}
    for i, gpu_id in enumerate(gpu_ids):
        # 균등 분배
        start_idx = i * (len(image_paths) // len(gpu_ids))
        if i == len(gpu_ids) - 1:
            end_idx = len(image_paths)  # 마지막 GPU는 남은 모든 이미지 처리
        else:
            end_idx = (i + 1) * (len(image_paths) // len(gpu_ids))
        
        images_per_gpu[gpu_id] = image_paths[start_idx:end_idx]
        print(f"GPU {gpu_id}: Assigned {len(images_per_gpu[gpu_id])} images")
    
    # 멀티프로세싱 설정
    multiprocessing.set_start_method('spawn', force=True)
    pool = multiprocessing.Pool(len(gpu_ids))
    
    # 각 GPU에 전달할 매개변수 준비
    worker_params = []
    for gpu_id, gpu_images in images_per_gpu.items():
        params = {
            'gpu_id': gpu_id,
            'image_paths': gpu_images,
            'output_dir': output_dir,
            'combined_base_dir': combined_base_dir,
            'config_path': config_path,
            'ckpt_path': ckpt_path,
            'guidance_scale': guidance_scale,
            'n_samples': n_samples,
            'ddim_steps': ddim_steps
        }
        worker_params.append(params)
    
    # 모든 GPU에 작업 할당
    results = []
    try:
        # 비동기 작업 제출
        async_results = pool.map_async(worker_process, worker_params)
        
        # 모든 결과 대기
        all_results = async_results.get()
        for batch_results in all_results:
            results.extend(batch_results)
    
    except Exception as e:
        print(f"Error in parallel processing: {e}")
    
    finally:
        pool.close()
        pool.join()
    
    # 결과 요약
    total_objects = sum(num_objects for _, num_objects in results)
    print(f"Processed {len(results)} images with a total of {total_objects} objects")
    print(f"Combined visualizations saved to {combined_base_dir}/combined and {combined_base_dir}/combined_masks")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Process images with amodal segmentation for all objects")
    parser.add_argument("--input_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--config", default="./configs/sd-finetune-pix2gestalt-c_concat-256.yaml", help="Config path")
    parser.add_argument("--ckpt", default="./ckpt/epoch=000005.ckpt", help="Checkpoint path")
    parser.add_argument("--gpu_ids", default=None, help="Comma-separated GPU IDs to use (e.g., '0,1,2,3')")
    parser.add_argument("--guidance_scale", type=float, default=2.0, help="Diffusion guidance scale")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate per object")
    parser.add_argument("--ddim_steps", type=int, default=200, help="Number of DDIM steps")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to process (default: all)")
    parser.add_argument("--combined_output_subdir", default="all_amodal", 
                        help="Subdirectory name for combined visualizations")
    
    args = parser.parse_args()
    
    # GPU IDs 파싱
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = args.gpu_ids
    
    # 병렬 처리 실행
    process_batch_parallel(
        args.input_dir,
        args.output_dir,
        args.config,
        args.ckpt,
        gpu_ids,
        args.guidance_scale,
        args.n_samples,
        args.ddim_steps,
        args.num_images,
        args.combined_output_subdir
    )

if __name__ == "__main__":
    main()