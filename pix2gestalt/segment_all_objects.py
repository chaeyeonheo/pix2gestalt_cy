import os
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import multiprocessing
from skimage.exposure import match_histograms
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# inference.py에서 필요한 함수들 임포트
from inference import (
    load_model_from_config,
    run_inference
)

class AmodalProcessor:
    def __init__(self, config_path, ckpt_path, device, sam_model_type='vit_h', verbose=False):
        """
        AmodalProcessor 초기화
        - Pix2Gestalt 모델과 SAM 모델을 로드하고
          자동 마스크 생성기를 설정합니다.
        """
        self.device = device
        self.verbose = verbose

        # Pix2Gestalt 모델 로드
        if verbose:
            print(f"[INFO] Pix2Gestalt 모델 로드: {ckpt_path} on {device}")
        config = OmegaConf.load(config_path)
        self.model = load_model_from_config(config, ckpt_path, device)

        # SAM 모델 로드 및 자동 마스크 생성기 초기화
        if verbose:
            print(f"[INFO] SAM 모델 로드 ({sam_model_type}) on {device}")
        sam_models = {
            'vit_b': './ckpt/sam_vit_b.pth',
            'vit_l': './ckpt/sam_vit_l.pth',
            'vit_h': './ckpt/sam_vit_h.pth'
        }
        sam = sam_model_registry[sam_model_type](checkpoint=sam_models[sam_model_type])
        sam.to(device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100
        )

    def generate_masks(self, image, white_mask=None):
        """
        SAM을 이용해 이미지의 모든 객체 마스크를 생성합니다.
        경계/비경계 구분 로직을 제거하고, 모든 마스크를 반환합니다.
        """
        if self.verbose:
            print("[INFO] 객체 마스크 생성 중...")
        masks = self.mask_generator.generate(image)
        if self.verbose:
            print(f"[INFO] 총 {len(masks)}개의 객체 마스크 생성됨")
        return masks, []

    def process_image_with_all_objects(self, image_path, output_dir, combined_base_dir,
                                      guidance_scale=2.0, n_samples=1, ddim_steps=200):
        """
        이미지 경로를 받아 SAM을 통해 모든 객체에 대해 Amodal Segmentation을 수행,
        결과를 합성해 최종 이미지를 생성합니다.
        """
        try:
            # 1) 이미지 로드 및 RGB 변환
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[ERROR] 이미지를 읽을 수 없음: {image_path}")
                return None, 0
                
            # 데이터 타입 확인 및 변환
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2) 리사이즈 및 흰색 마스크 생성
            target_size = 256
            image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)
            
            # 데이터 타입 확인 및 변환
            if image_resized.dtype != np.uint8:
                image_resized = image_resized.astype(np.uint8)
                
            white_mask = np.all(image_resized == 255, axis=2).astype(np.uint8)
            if self.verbose:
                print(f"[INFO] 흰색 마스크 영역 픽셀 수: {white_mask.sum()}")

            # 3) 객체 마스크 생성 (경계 구분 없이)
            all_masks, _ = self.generate_masks(image_resized, white_mask)

            # 4) 디렉토리 설정
            image_name = Path(image_path).stem
            image_output_dir = os.path.join(output_dir, image_name)
            os.makedirs(image_output_dir, exist_ok=True)
            combined_images_dir = os.path.join(combined_base_dir, "combined")
            combined_masks_dir = os.path.join(combined_base_dir, "combined_masks")
            debug_images_dir = os.path.join(combined_base_dir, "debug_images")
            debug_masks_dir = os.path.join(combined_base_dir, "debug_masks")
            os.makedirs(combined_images_dir, exist_ok=True)
            os.makedirs(combined_masks_dir, exist_ok=True)
            os.makedirs(debug_images_dir, exist_ok=True)
            os.makedirs(debug_masks_dir, exist_ok=True)

            # 5) 객체별 Amodal 추론 및 결과 저장 (히스토그램 매칭 제거)
            object_dirs = []
            for i, mask_data in enumerate(all_masks):
                # 마스크를 uint8로 명시적 변환
                mask_binary = mask_data['segmentation'].astype(np.uint8)
                mask_uint8 = mask_binary * 255
                
                object_dir = os.path.join(image_output_dir, f"object_{i:03d}")
                os.makedirs(object_dir, exist_ok=True)
                object_dirs.append(object_dir)
                
                # 마스크 저장
                cv2.imwrite(os.path.join(object_dir, "mask.png"), mask_uint8)

                # 마스킹된 입력 이미지 생성 및 저장
                masked_image = image_resized.copy()
                mask_3ch = np.stack([mask_binary]*3, axis=2)
                masked_image = np.where(mask_3ch == 1, masked_image, [255,255,255])
                
                # 데이터 타입 확인 및 변환
                if masked_image.dtype != np.uint8:
                    masked_image = masked_image.astype(np.uint8)
                    
                masked_input_path = os.path.join(object_dir, "masked_input.png")
                cv2.imwrite(
                    masked_input_path,
                    cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
                )

                # Amodal 추론 실행
                try:
                    preds = run_inference(
                        image_resized, mask_uint8, self.model,
                        guidance_scale, n_samples, ddim_steps,
                        device=self.device
                    )
                    for j, pred in enumerate(preds):
                        # int32 -> uint8 변환 후 BGR->RGB
                        pred_uint8 = pred.astype(np.uint8)
                        # 데이터 타입 확인 및 변환
                        if pred_uint8.dtype != np.uint8:
                            pred_uint8 = pred_uint8.astype(np.uint8)
                        pred_rgb = cv2.cvtColor(pred_uint8, cv2.COLOR_BGR2RGB)
                        
                        # 히스토그램 매칭 적용 - 원본 이미지의 색상 분포와 일치시킴
                        # 마스크 영역을 사용하여 객체 영역만 매칭
                        obj_mask = mask_data['segmentation'].astype(bool)
                        
                        # 마스크 영역 내의 원본 이미지 픽셀이 충분한지 확인 (최소 100픽셀)
                        if np.sum(obj_mask) > 100:
                            # 마스크 영역의 원본 이미지를 참조 이미지로 사용
                            masked_original = image_resized.copy()
                            masked_original[~obj_mask] = 255  # 배경을 흰색으로
                            
                            # 히스토그램 매칭 적용 (원본 객체 영역의 색상 분포를 유지)
                            matched_result = match_histograms(pred_rgb, masked_original, channel_axis=2)
                            
                            # 객체 영역 외의 픽셀은 원래 값으로 복원 (흰색)
                            white_bg = np.ones_like(pred_rgb) * 255
                            result_with_white_bg = np.where(np.stack([obj_mask]*3, axis=2), matched_result, white_bg)
                            
                            # 최종 결과 저장
                            save_path = os.path.join(object_dir, f"amodal_result_{j}.png")
                            cv2.imwrite(save_path, cv2.cvtColor(result_with_white_bg, cv2.COLOR_RGB2BGR))
                        else:
                            # 마스크 영역이 충분하지 않으면 히스토그램 매칭 없이 저장
                            save_path = os.path.join(object_dir, f"amodal_result_{j}.png")
                            cv2.imwrite(save_path, cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print(f"[ERROR] 객체 {i} 추론 실패: {e}")

            # 6) combined 단계: 결과 합성
            # 6) combined 단계: 결과 합성
            combined_vis = image_resized.copy()  # visible 영역은 유지

            for mask_data, object_dir in zip(all_masks, object_dirs):
                amodal_path = os.path.join(object_dir, "amodal_result_0.png")
                if not os.path.isfile(amodal_path):
                    continue

                amodal_bgr = cv2.imread(amodal_path)
                if amodal_bgr is None:
                    print(f"[WARNING] 파일을 읽을 수 없음: {amodal_path}")
                    continue

                amodal_rgb = cv2.cvtColor(amodal_bgr, cv2.COLOR_BGR2RGB)

                # 흰색이 아닌 픽셀 = 예측된 객체 영역
                obj_mask = ~np.all(amodal_rgb == [255, 255, 255], axis=2)

                # 업데이트 조건: 해당 픽셀이 amodal에서 예측되었고 AND white_mask에서 가려졌던 부분
                update_mask = obj_mask & (white_mask == 1)

                combined_vis[update_mask] = amodal_rgb[update_mask]

                if self.verbose:
                    print(f"[INFO] {object_dir} 적용됨: {np.sum(update_mask)} 픽셀 업데이트됨")



            # 7) 최종 결과 저장
            # 데이터 타입 확인 및 변환
            if combined_vis.dtype != np.uint8:
                combined_vis = combined_vis.astype(np.uint8)
                
            combined_path = os.path.join(combined_images_dir, f"{image_name}.png")
            cv2.imwrite(combined_path, cv2.cvtColor(combined_vis, cv2.COLOR_RGB2BGR))
            
            mask_path = os.path.join(combined_masks_dir, f"{image_name}.png")
            cv2.imwrite(mask_path, (combined_mask.astype(np.uint8) * 255))
            
            return image_name, len(all_masks)

        except Exception as e:
            print(f"[ERROR] process_image_with_all_objects 실패: {e}")
            return None, 0



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
            if image_name is not None:
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
    debug_images_dir = os.path.join(combined_base_dir, "debug_images")  # 디버깅용 이미지 폴더
    debug_masks_dir = os.path.join(combined_base_dir, "debug_masks")    # 디버깅용 마스크 폴더
    os.makedirs(combined_images_dir, exist_ok=True)
    os.makedirs(combined_masks_dir, exist_ok=True)
    os.makedirs(debug_images_dir, exist_ok=True)
    os.makedirs(debug_masks_dir, exist_ok=True)
    
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
            if batch_results:  # 결과가 있는 경우에만 추가
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
    parser = argparse.ArgumentParser(description="Process images with amodal segmentation focusing on mask boundaries")
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