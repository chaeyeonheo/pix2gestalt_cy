import os
import glob
import argparse
import torch
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from torchvision.transforms.functional import to_tensor
from multiprocessing import Pool, set_start_method, Manager
from math import ceil
from torch.cuda.amp import autocast  # 혼합 정밀도 계산을 위한 임포트
import time
from collections import defaultdict
from queue import Queue
from threading import Thread

from inference import run_inference, load_model_from_config, get_sam_predictor, run_sam
from segment_anything import SamAutomaticMaskGenerator
from masking_policies import box_mask, RandomMask, grid_probability_mask, uniform_grid_probability_mask, \
                              box_grid_probability_mask, mixed_mask, scatter_mask, center_box_mask, \
                              multiple_ratio_mask

# === 전역 변수 선언 ===
global_model = None
global_predictor = None
global_device = None

DEBUG_MODE = False  # --debug 플래그 여부를 반영

MASK_POLICIES = {
    'box': box_mask,
    'center_box': center_box_mask,
    'random': lambda shape, device, p: RandomMask(shape[-1]).to(device),
    'grid': grid_probability_mask,
    'uniform_grid': uniform_grid_probability_mask,
    'box_grid': box_grid_probability_mask,
    'scatter': scatter_mask,
    'mixed': mixed_mask,
    'multiple_ratio': multiple_ratio_mask,  # 새로 추가한 함수
}

def get_files_per_class(input_dir, split, limit_per_class=50):
    """각 클래스마다 지정된 개수의 이미지만 수집 - 더 깊은 폴더 구조 지원"""
    if split == 'train':
        # 클래스별로 이미지 수집
        files_by_class = {}
        total_classes = 0
        
        # 모든 깊이에서 이미지 파일 찾기
        for root, dirs, files in os.walk(input_dir):
            # 이미지 파일 필터링
            image_files = [os.path.join(root, f) for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                continue  # 이미지가 없으면 건너뛰기
                
            # 상대 경로 추출 (예: a/abbey 또는 a/abbey/subfolder)
            rel_path = os.path.relpath(root, input_dir)
            if rel_path == '.':
                continue  # 루트 디렉토리 건너뛰기
                
            # 경로 구성요소 분리
            path_components = rel_path.split(os.sep)
            
            # 경로가 충분히 깊을 경우 (최소 2단계: a/abbey)
            if len(path_components) >= 2:
                # 클래스 키 생성 (첫 두 수준만 사용: a/abbey)
                class_key = os.path.join(path_components[0], path_components[1])
                
                # 클래스 키가 아직 없으면 초기화
                if class_key not in files_by_class:
                    files_by_class[class_key] = []
                    total_classes += 1
                
                # 이미 충분한 이미지가 수집되었으면 스킵
                if len(files_by_class[class_key]) >= limit_per_class:
                    continue
                    
                # 필요한 만큼의 이미지만 추가
                remaining = limit_per_class - len(files_by_class[class_key])
                files_by_class[class_key].extend(image_files[:remaining])
                
                # 진행 상황 출력 (100개 클래스마다)
                if total_classes % 100 == 0 and len(files_by_class[class_key]) == limit_per_class:
                    print(f"처리 중: {total_classes}개 클래스 검색됨, 최근 클래스: {class_key}")
        
        # 결과 요약 출력
        print(f"총 {total_classes}개 클래스에서 이미지를 수집했습니다.")
        
        # 모든 클래스의 이미지를 하나의 리스트로 합치기
        all_files = []
        for class_key, files in files_by_class.items():
            print(f"클래스 {class_key}: {len(files)}개 이미지")
            all_files.extend(files)
            
        return sorted(all_files)
    
    elif split in ['val', 'test']:
        # 검증/테스트 세트는 클래스 구분 없이 전체 이미지 중 일부만 선택
        all_files = []
        for root, dirs, files in os.walk(input_dir):
            image_files = [os.path.join(root, f) for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
            all_files.extend(image_files)
            if len(all_files) >= limit_per_class:
                break
                
        return sorted(all_files[:limit_per_class])  # 테스트/검증 세트는 전체에서 일부만 선택
    
    else:
        raise ValueError(f"Unknown split: {split}")

def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)

def apply_mask(img, mask):
    return (img * mask).astype(np.uint8)

def visualize_mask_on_image(image, mask, alpha=0.5, color=(0,255,0)):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    mask_uint = mask.astype(np.uint8) if mask.dtype==bool else mask
    cm = np.zeros((*mask_uint.shape,3), dtype=np.uint8)
    cm[mask_uint>0] = color
    return cv2.addWeighted(image, 1, cm, alpha, 0)

def save_debug_image(img, path):
    if not DEBUG_MODE:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(img, np.ndarray):
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        if img.ndim == 2:
            rgb = np.stack([img]*3, axis=-1)
            Image.fromarray(rgb).save(path)
        else:
            Image.fromarray(img).save(path)
    else:
        img.save(path)

def init_process(model_config, ckpt, sam_model, device_id):
    global global_model, global_predictor, global_device
    global_device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    global_model = load_model_from_config(OmegaConf.load(model_config), ckpt, global_device)
    global_predictor = get_sam_predictor(model_type=sam_model, device=global_device)

def find_objects_in_image(img, orig_mask, name, debug_dir):
    global global_predictor
    global_predictor.set_image(img)
    h, w = img.shape[:2]

    def accept(m):
        m_sum = np.sum(m)
        if m_sum == 0:
            return False
        overlap = m & orig_mask
        overlap_ratio = np.sum(overlap) / m_sum
        return overlap_ratio < 0.8

    all_masks = []

    # 마스크 경계 포인트 활용
    ker = np.ones((5, 5), np.uint8)
    b = cv2.dilate(orig_mask.astype(np.uint8)*255, ker, 2) - cv2.erode(orig_mask.astype(np.uint8)*255, ker, 2)
    ys, xs = np.where(b > 0)
    pts = []
    for i in range(min(len(ys), 100)):
        y, x = ys[i], xs[i]
        for dy, dx in [(-3,0),(3,0),(0,-3),(0,3)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and not orig_mask[ny,nx]:
                pts.append(([nx, ny], 1))
    if pts:
        m, _ = run_sam(global_predictor, pts[:30])
        if m is not None and m.max()>0:
            mb = (m>0)
            save_debug_image(mb.astype(np.uint8)*255, os.path.join(debug_dir, 'boundary_mask.png'))
            all_masks.append(("boundary", mb))

    # 4분할 영역 중앙점 활용
    regions = [(0, 0, w//2, h//2), (w//2, 0, w, h//2), (0, h//2, w//2, h), (w//2, h//2, w, h)]
    for idx, (x1,y1,x2,y2) in enumerate(regions):
        cx, cy = (x1+x2)//2, (y1+y2)//2
        if not orig_mask[cy, cx]:
            m, _ = run_sam(global_predictor, [([cx, cy],1)])
            if m is not None and m.max()>0:
                mb = (m>0)
                save_debug_image(mb.astype(np.uint8)*255, os.path.join(debug_dir, f'region{idx}_mask.png'))
                all_masks.append((f"region{idx}", mb))

    # 중앙점 활용
    cx, cy = w//2, h//2
    if not orig_mask[cy, cx]:
        m, _ = run_sam(global_predictor, [([cx, cy],1)])
        if m is not None and m.max()>0:
            mb = (m>0)
            save_debug_image(mb.astype(np.uint8)*255, os.path.join(debug_dir, 'center_mask.png'))
            all_masks.append(("center", mb))

    # 격자 포인트 활용
    step = min(h, w)//8
    sparse_pts = []
    for y in range(step, h, step*2):
        for x in range(step, w, step*2):
            if not orig_mask[y, x]:
                sparse_pts.append(([x, y], 1))
    for i in range(0, len(sparse_pts), 5):
        batch = sparse_pts[i:i+5]
        if not batch:
            continue
        m, _ = run_sam(global_predictor, batch)
        if m is not None and m.max()>0:
            mb = (m>0)
            save_debug_image(mb.astype(np.uint8)*255, os.path.join(debug_dir, f'sparse{i}_mask.png'))
            all_masks.append((f"sparse{i}", mb))

    # 자동 마스크 생성기 활용
    auto_gen = SamAutomaticMaskGenerator(
        global_predictor.model,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        min_mask_region_area=100
    )
    all_anns = auto_gen.generate(img)
    for idx, ann in enumerate(all_anns, start=1):
        m = ann["segmentation"].astype(bool)
        save_debug_image(m.astype(np.uint8)*255, os.path.join(debug_dir, f'auto{idx}_mask.png'))
        all_masks.append((f"auto{idx}", m))

    # 모든 마스크에 대한 시각화
    all_masks_img = np.zeros_like(img)
    for i, (mask_name, mask) in enumerate(all_masks):
        color = [(i*50)%255, ((i*30)+100)%255, ((i*70)+150)%255]
        all_masks_img = visualize_mask_on_image(all_masks_img, mask, alpha=0.3, color=color)
    save_debug_image(all_masks_img, os.path.join(debug_dir, 'all_detected_masks.png'))

    # 필터 적용
    accepted_masks = []
    for mask_name, mask in all_masks:
        if accept(mask):
            accepted_masks.append(mask)

    # 필터 후 마스크 시각화
    if accepted_masks:
        accepted_img = np.zeros_like(img)
        for i, mask in enumerate(accepted_masks):
            color = [(i*50)%255, ((i*30)+100)%255, ((i*70)+150)%255]
            accepted_img = visualize_mask_on_image(accepted_img, mask, alpha=0.3, color=color)
        save_debug_image(accepted_img, os.path.join(debug_dir, 'accepted_masks.png'))
        
        # 원본 이미지 위에 수락된 마스크 오버레이
        overlay_img = img.copy()
        for i, mask in enumerate(accepted_masks):
            color = [(i*50)%255, ((i*30)+100)%255, ((i*70)+150)%255]
            overlay_img = visualize_mask_on_image(overlay_img, mask, alpha=0.3, color=color)
        save_debug_image(overlay_img, os.path.join(debug_dir, 'original_with_accepted_masks.png'))

    return accepted_masks

def get_output_path(img_path, out_root):
    """입력 이미지 경로에서 출력 경로 생성 - 알파벳/클래스/이미지ID 구조 유지"""
    # 입력 디렉토리 기준으로 상대 경로 추출
    # 예시 경로: ~/volum1/cy/latent-code-inpainting_new/data/data_large/data_256/a/abbey/00000001.jpg
    
    # 파일 및 디렉토리 구성요소 추출
    path_parts = img_path.split(os.sep)
    
    # 파일 이름 (확장자 제외)
    image_id = os.path.splitext(path_parts[-1])[0]
    
    # Places365 구조를 분석하여 알파벳(a) 및 클래스(abbey) 디렉토리 찾기
    # 파일 경로에서 역방향으로 검색
    alphabet_dir = None
    class_dir = None
    
    # 알파벳 디렉토리(a, b, c 등) 및 클래스 디렉토리(abbey, airport 등) 찾기
    for i in range(len(path_parts) - 2, 0, -1):
        if len(path_parts[i]) == 1 and path_parts[i].isalpha():  # 알파벳 디렉토리(a, b, c 등)
            alphabet_dir = path_parts[i]
            if i + 1 < len(path_parts) - 1:  # 다음 디렉토리가 클래스 디렉토리
                class_dir = path_parts[i + 1]
            break
    
    # 알파벳 디렉토리를 찾지 못한 경우, 마지막 두 수준을 사용
    if alphabet_dir is None and len(path_parts) >= 3:
        # 마지막 두 디렉토리 사용 (알파벳과 클래스로 가정)
        alphabet_dir = path_parts[-3]
        class_dir = path_parts[-2]
    
    # 알파벳과 클래스 디렉토리를 찾지 못한 경우, 기본값 사용
    if alphabet_dir is None:
        alphabet_dir = "unknown"
    if class_dir is None:
        class_dir = "unknown"
    
    # 디버그 정보 출력
    if DEBUG_MODE:
        print(f"입력 경로: {img_path}")
        print(f"알파벳 디렉토리: {alphabet_dir}")
        print(f"클래스 디렉토리: {class_dir}")
        print(f"이미지 ID: {image_id}")
    
    # 출력 경로 생성: [output_dir]/[alphabet]/[class]/[image_id]/
    out_dir = os.path.join(out_root, alphabet_dir, class_dir, image_id)
    
    return out_dir

# === 배치 처리 및 혼합 정밀도를 적용한 새로운 함수 ===
# 배치 처리 함수에 진행 상황 로깅 추가
def process_batch(batch_paths, out_root, mask_policy, mask_prob, multiple_mask_policies, gs, steps):
    """배치로 이미지 처리 - 혼합 정밀도 적용"""
    global global_model, global_predictor, global_device
    
    start_time = time.time()
    
    # 출력 경로 생성
    output_dirs = [get_output_path(img_path, out_root) for img_path in batch_paths]
    debug_dirs = [os.path.join(out_dir, "debug") for out_dir in output_dirs]
    
    # 디렉토리 생성
    for i, (out_dir, debug_dir) in enumerate(zip(output_dirs, debug_dirs)):
        os.makedirs(out_dir, exist_ok=True)
        if DEBUG_MODE:
            os.makedirs(debug_dir, exist_ok=True)
    
    # 이미지 로드 및 텐서 변환
    batch_imgs = []
    batch_tensors = []
    
    for img_path in batch_paths:
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        batch_imgs.append(img)
        # 텐서로 변환 (GPU로 이동은 나중에 일괄적으로)
        batch_tensors.append(to_tensor(img).unsqueeze(0))
    
    # 텐서들을 하나의 배치로 결합
    if batch_tensors:
        try:
            # 배치 텐서 생성 및 GPU로 이동
            batch_tensor = torch.cat(batch_tensors, dim=0).to(global_device)
            
            # 혼합 정밀도 적용
            with autocast():
                # 마스크 생성 - 각 이미지마다 다른 마스크 적용
                batch_masks = []
                for i in range(len(batch_paths)):
                    img_tensor = batch_tensor[i:i+1]  # 배치에서 단일 이미지 추출
                    
                    # 마스크 생성
                    if mask_policy == 'multiple_ratio' and multiple_mask_policies:
                        mask_tensor = multiple_ratio_mask(img_tensor.shape, global_device, multiple_mask_policies)
                    else:
                        mask_tensor = MASK_POLICIES[mask_policy](img_tensor.shape, global_device, p=mask_prob).float()
                    
                    batch_masks.append(mask_tensor)
                
                # 각 이미지 개별 처리
                for i, (img, img_path, mask_tensor, out_dir, debug_dir) in enumerate(
                        zip(batch_imgs, batch_paths, batch_masks, output_dirs, debug_dirs)):
                    
                    # 마스크를 numpy로 변환
                    mask_np = mask_tensor[0, 0].cpu().numpy()
                    mask_img = (mask_np * 255).astype(np.uint8)
                    masked_img = apply_mask(img, mask_np[:, :, None])
                    
                    # 이미지 저장
                    image_id = os.path.splitext(os.path.basename(img_path))[0]
                    save_image(os.path.join(out_dir, 'original.png'), img)
                    save_image(os.path.join(out_dir, 'mask.png'), mask_img)
                    save_image(os.path.join(out_dir, 'partial.png'), masked_img)
                    
                    # 마스크 반전 - SAM은 0이 마스크된 영역, 1이 배경임
                    orig_mask = mask_img == 0
                    
                    # 객체 찾기
                    objs = find_objects_in_image(img, orig_mask, image_id, debug_dir)
                    
                    # 각 객체에 대한 completion 생성 - 혼합 정밀도 적용
                    for j, obj_mask in enumerate(objs, 1):
                        visible_mask = (obj_mask * 255).astype(np.uint8)
                        with autocast():
                            result = run_inference(img, visible_mask, global_model, gs, 1, steps, global_device)
                        if result and result[0] is not None:
                            save_image(os.path.join(out_dir, f'comp{j}.png'), result[0])
                            save_debug_image(obj_mask * 255, os.path.join(debug_dir, f'comp_mask_{j}.png'))
        
        except RuntimeError as e:
            # CUDA 메모리 오류 시 배치 크기를 줄여서 개별 처리
            print(f"배치 처리 중 오류 발생: {e}")
            print("이미지를 개별적으로 처리합니다...")
            
            for img_path in batch_paths:
                try:
                    # 단일 이미지 처리
                    out_dir = get_output_path(img_path, out_root)
                    debug_dir = os.path.join(out_dir, "debug")
                    
                    os.makedirs(out_dir, exist_ok=True)
                    if DEBUG_MODE:
                        os.makedirs(debug_dir, exist_ok=True)
                    
                    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    img_tensor = to_tensor(img).unsqueeze(0).to(global_device)
                    
                    with autocast():
                        if mask_policy == 'multiple_ratio' and multiple_mask_policies:
                            mask_tensor = multiple_ratio_mask(img_tensor.shape, global_device, multiple_mask_policies)
                        else:
                            mask_tensor = MASK_POLICIES[mask_policy](img_tensor.shape, global_device, p=mask_prob).float()
                        
                        mask_np = mask_tensor[0, 0].cpu().numpy()
                        mask_img = (mask_np * 255).astype(np.uint8)
                        masked_img = apply_mask(img, mask_np[:, :, None])
                        
                        image_id = os.path.splitext(os.path.basename(img_path))[0]
                        save_image(os.path.join(out_dir, 'original.png'), img)
                        save_image(os.path.join(out_dir, 'mask.png'), mask_img)
                        save_image(os.path.join(out_dir, 'partial.png'), masked_img)
                        
                        orig_mask = mask_img == 0
                        objs = find_objects_in_image(img, orig_mask, image_id, debug_dir)
                        
                        for j, obj_mask in enumerate(objs, 1):
                            visible_mask = (obj_mask * 255).astype(np.uint8)
                            result = run_inference(img, visible_mask, global_model, gs, 1, steps, global_device)
                            if result and result[0] is not None:
                                save_image(os.path.join(out_dir, f'comp{j}.png'), result[0])
                                save_debug_image(obj_mask * 255, os.path.join(debug_dir, f'comp_mask_{j}.png'))
                except Exception as e:
                    print(f"이미지 {img_path} 처리 중 오류: {e}")
    
    end_time = time.time()
    print(f"배치 {len(batch_paths)}개 처리 완료, 소요 시간: {end_time - start_time:.2f}초")


def worker_main(batch, out_root, model_config, ckpt, sam_model, 
               mask_policy, mask_prob, multiple_mask_policies, gs, steps, 
               device_id, batch_size, total_images, start_idx):
    """배치 크기로 나누어 작업을 처리하는 워커 함수"""
    init_process(model_config, ckpt, sam_model, device_id)
    
    # 전체 진행 상황을 위한 카운터
    processed_count = 0
    
    # 배치 크기로 나누기
    for i in range(0, len(batch), batch_size):
        sub_batch = batch[i:i+batch_size]
        try:
            process_batch(sub_batch, out_root, mask_policy, mask_prob, 
                        multiple_mask_policies, gs, steps)
            
            # 진행 상황 업데이트
            processed_count += len(sub_batch)
            current_idx = start_idx + processed_count
            progress_percent = (current_idx / total_images) * 100
            print(f"GPU{device_id}: {current_idx}/{total_images} 완료 ({progress_percent:.2f}%), 남은 이미지: {total_images - current_idx}")
            
        except Exception as e:
            print(f"GPU{device_id}에서 배치 처리 중 오류: {e}")
            # 오류가 발생한 경우 한 장씩 처리
            for img_path in sub_batch:
                try:
                    process_batch([img_path], out_root, mask_policy, mask_prob, 
                                multiple_mask_policies, gs, steps)
                    
                    # 단일 이미지 처리 성공 시 진행 상황 업데이트
                    processed_count += 1
                    current_idx = start_idx + processed_count
                    progress_percent = (current_idx / total_images) * 100
                    print(f"GPU{device_id}: {current_idx}/{total_images} 완료 ({progress_percent:.2f}%), 남은 이미지: {total_images - current_idx}")
                    
                except Exception as e2:
                    print(f"이미지 {img_path} 처리 중 오류: {e2}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='입력 이미지 디렉토리')
    parser.add_argument('--output_dir', required=True, help='출력 디렉토리')
    parser.add_argument('--model_config', required=True, help='모델 구성 파일 경로')
    parser.add_argument('--model_ckpt', required=True, help='모델 체크포인트 파일 경로')
    parser.add_argument('--mask_policy', default='box', 
                        choices=['box', 'center_box', 'random', 'grid', 'uniform_grid', 
                                'box_grid', 'scatter', 'mixed', 'multiple_ratio'],
                        help='마스킹 정책')
    parser.add_argument('--mask_prob', type=float, default=0.5, help='마스크 확률/크기')
    parser.add_argument('--multiple_mask_policies', type=str, default=None,
                       help='다중 마스킹 정책 설정 (JSON 형식). 예: [["box",0.5,0.4],["center_box",0.8,0.3],["random",0.3,0.3]]')
    parser.add_argument('--guidance_scale', type=float, default=2.0, help='가이던스 스케일')
    parser.add_argument('--ddim_steps', type=int, default=50, help='DDIM 스텝 수')
    parser.add_argument('--sam_model', default='vit_h', help='SAM 모델 타입 (vit_b, vit_l, vit_h)')
    parser.add_argument('--num_gpus', type=int, default=1, help='사용할 GPU 수')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val', help='데이터셋 분할')
    parser.add_argument('--debug', action='store_true', help='디버그 모드 활성화')
    parser.add_argument('--limit_per_class', type=int, default=50, help='각 클래스마다 처리할 최대 이미지 수')
    parser.add_argument('--batch_size', type=int, default=4, help='배치 크기')
    args = parser.parse_args()

    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    # 사용자가 지정한 배치 크기 그대로 사용
    batch_size = args.batch_size
    
    print(f"사용 중인 SAM 모델: {args.sam_model}, 배치 크기: {batch_size}")
    
    # 다중 마스킹 정책 처리
    multiple_mask_policies = None
    if args.multiple_mask_policies:
        multiple_mask_policies = json.loads(args.multiple_mask_policies)
    elif args.mask_policy == 'multiple_ratio':
        multiple_mask_policies = [
            ["box", 0.5, 0.4],
            ["center_box", 0.8, 0.3],
            ["random", 0.3, 0.3]
        ]
        print(f"기본 다중 마스킹 정책 사용: {multiple_mask_policies}")
    
    # 처리 시간 측정 시작
    start_time = time.time()
    
    # 파일 목록 가져오기
    files = get_files_per_class(args.input_dir, args.split, args.limit_per_class)
    total_images = len(files)
    print(f"총 {total_images}개 이미지를 처리합니다.")
    
    if total_images == 0:
        print(f"경고: '{args.input_dir}'에서 이미지를 찾을 수 없습니다!")
        return
    
    # GPU 분배
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"사용 가능한 GPU: {torch.cuda.device_count()}개, 실제 사용: {num_gpus}개")
    
    # 인터리빙 방식으로 파일 분배 (라운드 로빈 방식)
    batches = [[] for _ in range(num_gpus)]
    for i, file in enumerate(files):
        gpu_idx = i % num_gpus
        batches[gpu_idx].append(file)
    
    # 각 GPU에 할당된 파일 수 출력 및 시작 인덱스 계산
    start_indices = [0] * num_gpus
    for i in range(1, num_gpus):
        start_indices[i] = start_indices[i-1] + len(batches[i-1])
    
    for i, (batch, start_idx) in enumerate(zip(batches, start_indices)):
        print(f"GPU {i}: {len(batch)}개 이미지 처리 예정 (시작 인덱스: {start_idx})")

    # 출력 디렉토리 생성 - 진행 상황 파일 저장을 위해
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 진행 상황 파일 초기화
    progress_file = os.path.join(args.output_dir, "processing_progress.txt")
    with open(progress_file, "w") as f:
        f.write(f"처리 시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"총 이미지 수: {total_images}\n")
        f.write("-" * 50 + "\n")

    # 멀티프로세싱 설정 및 실행
    set_start_method('spawn', force=True)
    with Pool(processes=num_gpus) as pool:
        pool.starmap(
            worker_main,
            [(batch, args.output_dir, args.model_config, args.model_ckpt, args.sam_model,
              args.mask_policy, args.mask_prob, multiple_mask_policies, args.guidance_scale, 
              args.ddim_steps, gpu_id, batch_size, total_images, start_idx)
             for gpu_id, (batch, start_idx) in enumerate(zip(batches, start_indices))]
        )
    
    # 처리 시간 측정 종료 및 출력
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 최종 진행 상황 업데이트
    completion_info = (
        f"\n처리 완료!\n"
        f"총 {total_images}개 이미지 처리 완료\n"
        f"총 소요 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.2f}초\n"
        f"이미지당 평균 처리 시간: {total_time/total_images:.2f}초\n"
        f"처리 종료 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    print(completion_info)
    
    # 진행 상황 파일에 최종 정보 추가
    with open(progress_file, "a") as f:
        f.write(completion_info)

if __name__ == '__main__':
    main()