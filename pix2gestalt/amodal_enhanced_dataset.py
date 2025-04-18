import numpy as np
import torch
import cv2
import os
import sys
import time
import glob
from PIL import Image
from tqdm import tqdm
import gc
from omegaconf import OmegaConf
import multiprocessing as mp
import argparse
import math
import uuid
import matplotlib.pyplot as plt

# inference.py에서 필요한 함수들 가져오기
from inference import load_model_from_config, get_sam_predictor, run_sam, run_inference

def save_debug_image(img, path):
    """디버깅용 이미지 저장 함수"""
    if isinstance(img, np.ndarray):
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        if len(img.shape) == 2:  # 그레이스케일 마스크인 경우
            # 3채널 RGB로 변환
            rgb_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            rgb_img[:, :, 0] = img
            rgb_img[:, :, 1] = img
            rgb_img[:, :, 2] = img
            Image.fromarray(rgb_img).save(path)
        else:
            Image.fromarray(img).save(path)
    else:
        img.save(path)

def visualize_mask_on_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    """마스크를 이미지 위에 시각화하는 함수"""
    # 이미지와 마스크가 NumPy 배열인지 확인
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    
    # 마스크가 불리언 타입이면 0과 1로 변환
    if mask.dtype == bool:
        mask = mask.astype(np.uint8)
    
    # 마스크가 1채널이면 3채널로 변환
    if len(mask.shape) == 2:
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_mask[mask > 0] = color
    else:
        color_mask = mask
    
    # 합성 이미지 생성
    blended = cv2.addWeighted(image, 1, color_mask, alpha, 0)
    return blended

def generate_grid_points(mask, h, w, grid_size=16):
    """그리드 포인트 생성 함수"""
    points = []
    for y in range(grid_size, h, grid_size):
        for x in range(grid_size, w, grid_size):
            # 가시적인 영역의 포인트만 사용
            if not mask[y, x]:
                points.append(([x, y], 1))  # 전경 포인트
    return points

def process_batch(args):
    """
    GPU 한 개에서 처리할 배치를 처리하는 함수
    """
    batch_files, input_dir, output_dir, model_config_path, model_ckpt_path, sam_model_type, device_id, guidance_scale, ddim_steps, grid_size = args
    
    # 디버깅 디렉토리 설정
    debug_dir = os.path.join(output_dir, 'debug')
    sam_masks_dir = os.path.join(debug_dir, 'sam_masks')
    completion_dir = os.path.join(debug_dir, 'completions')
    overlay_dir = os.path.join(debug_dir, 'overlays')
    filled_masks_dir = os.path.join(output_dir, 'filled_masks')  # 최종 채워진 마스크를 저장할 별도 폴더
    
    # 필요한 디렉토리 생성
    os.makedirs(sam_masks_dir, exist_ok=True)
    os.makedirs(completion_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(filled_masks_dir, exist_ok=True)
    
    # 현재 프로세스에 GPU 디바이스 할당
    device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    print(f"프로세스 시작: GPU {device_id}, 처리할 이미지 수: {len(batch_files)}")
    
    # 모델 로드
    config = OmegaConf.load(model_config_path)
    model = load_model_from_config(config, model_ckpt_path, device)
    
    # SAM 예측기 가져오기
    sam_predictor = get_sam_predictor(model_type=sam_model_type, device=device)
    
    # 각 이미지 처리
    for img_file in tqdm(batch_files, desc=f"GPU {device_id} 처리 중"):
        img_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        filled_mask_path = os.path.join(filled_masks_dir, f"{os.path.splitext(img_file)[0]}_filled_mask.png")
        
        try:
            # 이미지 로드
            input_image = cv2.imread(img_path)
            if input_image is None:
                print(f"이미지를 불러올 수 없습니다: {img_path}")
                continue
                
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            
            # 마스크 생성 (흰색/검은색 영역이 마스킹됨을 가정)
            white_mask = np.all(input_image == 255, axis=2)
            black_mask = np.all(input_image == 0, axis=2)
            mask = white_mask | black_mask
            
            # 이진 마스크로 변환 (마스킹된 영역은 1, 가시적인 영역은 0)
            binary_mask = mask.astype(np.uint8) * 255
            
            # 디버깅: 원본 마스크 저장
            mask_debug_path = os.path.join(debug_dir, f"{os.path.splitext(img_file)[0]}_original_mask.png")
            save_debug_image(binary_mask, mask_debug_path)
            
            # 마스크가 없으면 다음 이미지로
            if binary_mask.sum() == 0:
                print(f"{img_file}에 마스크가 없습니다. 원본 이미지를 복사합니다.")
                output_img = Image.fromarray(input_image)
                output_img.save(output_path)
                continue
            
            # SAM 이미지 설정
            sam_predictor.set_image(input_image)
            
            # 결과 이미지 (입력 이미지로 초기화)
            result_image = input_image.copy()
            
            # 채워진 마스크 픽셀 추적
            filled_mask = np.zeros_like(binary_mask)
            
            # 그리드 포인트 생성 (더 촘촘하게)
            h, w = input_image.shape[:2]
            points = generate_grid_points(mask, h, w, grid_size)
            
            # 테두리에 더 많은 포인트 추가 (마스크 근처에서의 감지를 향상)
            kernel = np.ones((5, 5), np.uint8)
            mask_boundary = cv2.dilate(binary_mask, kernel, iterations=2) - cv2.erode(binary_mask, kernel, iterations=2)
            
            # 마스크 경계 근처의 가시적인 영역에 포인트 추가
            boundary_points = []
            ys, xs = np.where(mask_boundary > 0)
            for i in range(len(ys)):
                y, x = ys[i], xs[i]
                # 마스크 경계 주변의 가시적인 영역에 포인트 추가
                for dy in [-3, -2, -1, 0, 1, 2, 3]:
                    for dx in [-3, -2, -1, 0, 1, 2, 3]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx]:
                            boundary_points.append(([nx, ny], 1))
            
            # 중복 제거 및 포인트 추가
            unique_boundary_points = []
            boundary_coords = set()
            for point, label in boundary_points:
                coord = (point[0], point[1])
                if coord not in boundary_coords:
                    boundary_coords.add(coord)
                    unique_boundary_points.append((point, label))
            
            points.extend(unique_boundary_points)
            
            if len(points) == 0:
                print(f"{img_file}에 가시적인 영역이 없습니다. 원본 이미지를 복사합니다.")
                output_img = Image.fromarray(input_image)
                output_img.save(output_path)
                continue
            
            print(f"SAM 프롬프트용 {len(points)}개 포인트 생성됨 (이미지: {img_file})")
            
            # 메모리 문제를 피하기 위해 작은 배치로 처리
            batch_points = [points[i:i+10] for i in range(0, len(points), 10)]
            
            # 객체 인덱스 추적
            object_idx = 0
            has_successful_completion = False
            
            for point_batch in batch_points:
                if not point_batch:
                    continue
                    
                try:
                    # 이 배치의 포인트에 대한 가시적 마스크 가져오기
                    visible_mask, overlay_masks = run_sam(sam_predictor, point_batch)
                    
                    # 유효한 마스크가 생성되지 않으면 건너뜀
                    if visible_mask is None or visible_mask.max() == 0:
                        continue
                    
                    # 객체 인덱스 증가
                    object_idx += 1
                    
                    # 디버깅: SAM 마스크 저장
                    sam_mask_path = os.path.join(sam_masks_dir, f"{os.path.splitext(img_file)[0]}_obj{object_idx}_mask.png")
                    save_debug_image(visible_mask, sam_mask_path)
                    
                    # SAM 마스크 오버레이 저장
                    overlay_img = visualize_mask_on_image(input_image, visible_mask > 0)
                    overlay_path = os.path.join(overlay_dir, f"{os.path.splitext(img_file)[0]}_obj{object_idx}_overlay.png")
                    save_debug_image(overlay_img, overlay_path)
                    
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
                        
                        # 디버깅: 완성된 객체 저장
                        completion_path = os.path.join(completion_dir, f"{os.path.splitext(img_file)[0]}_obj{object_idx}_completion.png")
                        save_debug_image(completion, completion_path)
                        
                        # 마스크된 영역만 업데이트
                        update_mask = np.logical_and(visible_mask > 0, binary_mask > 0)
                        update_mask = np.logical_and(update_mask, ~filled_mask)
                        
                        # 디버깅: 업데이트 마스크 저장
                        update_mask_path = os.path.join(debug_dir, f"{os.path.splitext(img_file)[0]}_obj{object_idx}_update_mask.png")
                        save_debug_image(update_mask.astype(np.uint8) * 255, update_mask_path)
                        
                        # 마스크된 영역 업데이트
                        if np.any(update_mask):
                            # 업데이트 마스크 영역에만 완성된 이미지 적용
                            for c in range(3):  # RGB 채널
                                result_image[:,:,c] = np.where(update_mask, completion[:,:,c], result_image[:,:,c])
                            
                            # 채워진 마스크 업데이트
                            filled_mask = np.logical_or(filled_mask, update_mask)
                            has_successful_completion = True
                            
                            # 디버깅: 현재까지의 결과 저장
                            interim_result_path = os.path.join(debug_dir, f"{os.path.splitext(img_file)[0]}_interim_result_obj{object_idx}.png")
                            save_debug_image(result_image, interim_result_path)
                    
                except Exception as e:
                    print(f"포인트 배치 처리 중 오류 발생: {e}")
                    continue
                
                # 메모리 정리
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 채워진 비율 계산
            fill_ratio = np.sum(filled_mask) / np.sum(binary_mask) * 100
            
            # 디버깅: 최종 채워진 마스크를 별도 폴더에 저장
            save_debug_image(filled_mask.astype(np.uint8) * 255, filled_mask_path)
            
            # 결과 이미지 항상 저장 (충분히 채워지지 않았더라도)
            output_img = Image.fromarray(result_image)
            output_img.save(output_path)
            
            print(f"GPU {device_id}: {img_file} 처리 완료. 마스크 영역의 {fill_ratio:.2f}% 채워짐")
            
        except Exception as e:
            print(f"이미지 {img_file} 처리 중 오류 발생: {e}")
            continue
    
    # 최종 정리
    del model, sam_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return True


def create_amodal_enhanced_dataset_distributed(
    input_dir,           # 마스킹된 이미지가 있는 디렉토리
    output_dir,          # 향상된 이미지를 저장할 디렉토리
    model_config_path=None,   # pix2gestalt 모델 설정 경로
    model_ckpt_path=None,     # pix2gestalt 모델 체크포인트 경로
    sam_model_type='vit_h', # SAM 모델 타입
    guidance_scale=2.0,  # 확산 가이던스 스케일
    ddim_steps=50,       # 확산 스텝 수
    grid_size=16,        # 그리드 포인트의 간격
    num_gpus=8,          # 사용할 GPU 수
    max_images=None      # 처리할 최대 이미지 수 (None이면 모든 이미지 처리)
):
    # 기본 경로 설정 (제공되지 않은 경우)
    if model_config_path is None:
        model_config_path = './configs/sd-finetune-pix2gestalt-c_concat-256.yaml' # 기본 설정 파일 경로
    if model_ckpt_path is None:
        model_ckpt_path = './ckpt/epoch=000005.ckpt'  # 기본 체크포인트 경로
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 디버깅 디렉토리 생성
    debug_dir = os.path.join(output_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(os.path.join(debug_dir, 'sam_masks'), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, 'completions'), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, 'overlays'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'filled_masks'), exist_ok=True)  # 최종 채워진 마스크를 저장할 별도 폴더
    
    # 사용할 GPU 수 확인
    available_gpus = torch.cuda.device_count()
    if available_gpus < num_gpus:
        print(f"요청한 GPU 수({num_gpus})가 사용 가능한 GPU 수({available_gpus})보다 많습니다. 사용 가능한 모든 GPU를 사용합니다.")
        num_gpus = available_gpus
    
    if num_gpus == 0:
        print("사용 가능한 GPU가 없습니다. CPU를 사용합니다.")
        num_gpus = 1  # CPU 모드
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # 처리할 이미지 수 제한
    if max_images is not None and max_images > 0 and max_images < len(image_files):
        image_files = image_files[:max_images]
        print(f"전체 {len(os.listdir(input_dir))}개 이미지 중 {max_images}개만 처리합니다.")
    
    total_images = len(image_files)
    print(f"총 {total_images}개 이미지를 처리합니다.")
    
    if total_images == 0:
        print("처리할 이미지가 없습니다.")
        return
    
    # 각 GPU가 처리할 이미지 배치 생성
    images_per_gpu = math.ceil(total_images / num_gpus)
    batches = []
    
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * images_per_gpu
        end_idx = min(start_idx + images_per_gpu, total_images)
        
        if start_idx >= total_images:
            break
            
        batch_files = image_files[start_idx:end_idx]
        batches.append((
            batch_files, 
            input_dir, 
            output_dir, 
            model_config_path, 
            model_ckpt_path, 
            sam_model_type, 
            gpu_id, 
            guidance_scale, 
            ddim_steps,
            grid_size
        ))
    
    # 프로세스 풀 생성 및 배치 처리 시작
    print(f"{num_gpus}개의 GPU로 분산 처리를 시작합니다.")
    start_time = time.time()
    
    # 멀티프로세싱 풀 생성 및 배치 처리
    with mp.Pool(processes=num_gpus) as pool:
        results = pool.map(process_batch, batches)
    
    elapsed_time = time.time() - start_time
    print(f"모든 처리 완료: {elapsed_time:.2f}초 소요됨")


def main():
    parser = argparse.ArgumentParser(description='아모달 세그멘테이션을 활용한 인페인팅 데이터셋 생성 (다중 GPU 버전)')
    parser.add_argument('--input_dir', type=str, required=True, help='마스킹된 이미지가 있는 디렉토리')
    parser.add_argument('--output_dir', type=str, required=True, help='향상된 이미지를 저장할 디렉토리')
    parser.add_argument('--model_config', type=str, default='./configs/sd-finetune-pix2gestalt-c_concat-256.yaml', help='pix2gestalt 모델 설정 경로')
    parser.add_argument('--model_ckpt', type=str, default='./ckpt/epoch=000005.ckpt', help='pix2gestalt 모델 체크포인트 경로')
    parser.add_argument('--sam_model', type=str, default='vit_h', help='SAM 모델 타입 (vit_b, vit_l, vit_h)')
    parser.add_argument('--guidance_scale', type=float, default=2.0, help='확산 가이던스 스케일')
    parser.add_argument('--ddim_steps', type=int, default=50, help='확산 스텝 수')
    parser.add_argument('--grid_size', type=int, default=16, help='그리드 포인트의 간격 (작을수록 더 많은 포인트 생성)')
    parser.add_argument('--num_gpus', type=int, default=8, help='사용할 GPU 수')
    parser.add_argument('--max_images', type=int, default=None, help='처리할 최대 이미지 수 (기본값: 모든 이미지)')
    args = parser.parse_args()
    
    create_amodal_enhanced_dataset_distributed(
        args.input_dir,
        args.output_dir,
        args.model_config,
        args.model_ckpt,
        args.sam_model,
        args.guidance_scale,
        args.ddim_steps,
        args.grid_size,
        args.num_gpus,
        args.max_images
    )

if __name__ == "__main__":
    # 멀티프로세싱 시작 방법 설정 (윈도우에서 필요)
    mp.set_start_method('spawn', force=True)
    main()