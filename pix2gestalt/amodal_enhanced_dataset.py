import os
import sys
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import gc
import shutil
from omegaconf import OmegaConf
import multiprocessing as mp

# 필요한 모듈 임포트
from inference import load_model_from_config, get_sam_predictor, run_sam, run_inference
from segment_anything import SamAutomaticMaskGenerator

def save_debug_image(img, path):
    """디버그 이미지를 저장하는 헬퍼 함수"""
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

def visualize_mask_on_image(image, mask, alpha=0.5, color=(0,255,0)):
    """마스크를 이미지 위에 시각화하는 함수"""
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    mask_uint = mask.astype(np.uint8) if mask.dtype==bool else mask
    cm = np.zeros((*mask_uint.shape,3), dtype=np.uint8)
    cm[mask_uint>0] = color
    return cv2.addWeighted(image, 1, cm, alpha, 0)

def find_objects_in_image(sam_predictor, img, orig_mask, name, debug_dir):
    """
    SAM으로부터 객체 후보 마스크를 뽑아오는 함수
    1) manual 전략: boundary / 4-quadrant / center / sparse
    2) automatic mask generator
    3) 마스크 영역과 80% 이상 겹치면 제외 (그 외 모든 마스크 허용)
    """
    h, w = img.shape[:2]
    objs = []

    def accept(m):
        """
        마스크 필터링 함수:
        마스크 영역과 80% 이상 겹치면 제외 (마스크 자체를 객체로 인식한 경우)
        그 외 모든 마스크는 허용
        """
        # 마스크와 겹치는 비율 계산
        m_sum = np.sum(m)
        if m_sum == 0:
            return False
            
        overlap = m & orig_mask
        overlap_ratio = np.sum(overlap) / m_sum
        
        # 마스크 영역과 80% 이상 겹치면 제외 (나머지는 모두 허용)
        return overlap_ratio < 0.8

    # 모든 전략의 마스크를 저장할 리스트
    all_masks = []
    
    # 1) Boundary-based points
    ker = np.ones((5, 5), np.uint8)
    b = cv2.dilate(orig_mask.astype(np.uint8)*255, ker, 2) \
      - cv2.erode (orig_mask.astype(np.uint8)*255, ker, 2)
    ys, xs = np.where(b > 0)
    pts = []
    for i in range(min(len(ys), 100)):
        y, x = ys[i], xs[i]
        for dy, dx in [(-3,0),(3,0),(0,-3),(0,3)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and not orig_mask[ny,nx]:
                pts.append(([nx, ny], 1))
    if pts:
        m, _ = run_sam(sam_predictor, pts[:30])
        if m is not None and m.max()>0:
            mb = (m>0)
            print(f"[DEBUG][{name}] boundary SAM mask size = {mb.sum()}")
            save_debug_image(mb.astype(np.uint8)*255,
                             os.path.join(debug_dir, 'sam_masks', f"{name}_boundary.png"))
            all_masks.append(("boundary", mb))

    # 2) 4-Quadrant centers
    regions = [
        (0, 0, w//2, h//2),
        (w//2, 0, w, h//2),
        (0, h//2, w//2, h),
        (w//2, h//2, w, h)
    ]
    for idx, (x1,y1,x2,y2) in enumerate(regions):
        cx, cy = (x1+x2)//2, (y1+y2)//2
        if not orig_mask[cy, cx]:
            m, _ = run_sam(sam_predictor, [([cx, cy],1)])
            if m is not None and m.max()>0:
                mb = (m>0)
                print(f"[DEBUG][{name}] region{idx}@({cx},{cy}) SAM mask size = {mb.sum()}")
                save_debug_image(mb.astype(np.uint8)*255,
                                 os.path.join(debug_dir, 'sam_masks', f"{name}_reg{idx}_{cx}_{cy}.png"))
                all_masks.append((f"region{idx}", mb))

    # 3) Image center
    cx, cy = w//2, h//2
    if not orig_mask[cy, cx]:
        m, _ = run_sam(sam_predictor, [([cx, cy],1)])
        if m is not None and m.max()>0:
            mb = (m>0)
            print(f"[DEBUG][{name}] center@({cx},{cy}) SAM mask size = {mb.sum()}")
            save_debug_image(mb.astype(np.uint8)*255,
                             os.path.join(debug_dir, 'sam_masks', f"{name}_center.png"))
            all_masks.append(("center", mb))

    # 4) Sparse grid sampling
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
        m, _ = run_sam(sam_predictor, batch)
        if m is not None and m.max()>0:
            mb = (m>0)
            print(f"[DEBUG][{name}] sparse#{i} SAM mask size = {mb.sum()}")
            save_debug_image(mb.astype(np.uint8)*255,
                             os.path.join(debug_dir, 'sam_masks', f"{name}_sparse_{i}.png"))
            all_masks.append((f"sparse{i}", mb))

    # 5) AutomaticMaskGenerator로 이미지 전체 탐색
    auto_gen = SamAutomaticMaskGenerator(
        sam_predictor.model,
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
        print(f"[DEBUG][{name}] auto#{idx} mask size = {m.sum()}")
        save_debug_image(m.astype(np.uint8)*255,
                         os.path.join(debug_dir, 'sam_masks', f"{name}_auto_{idx}.png"))
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
            m_sum = np.sum(mask)
            overlap = mask & orig_mask
            overlap_ratio = np.sum(overlap) / m_sum if m_sum > 0 else 0
            print(f"[DEBUG][{name}] Accepted {mask_name}, overlap_ratio={overlap_ratio:.2f}")
            
            # 마스크 영역과 겹치는 부분 있는지 확인 (디버깅용)
            has_overlap = np.any(overlap)
            if has_overlap:
                print(f"[DEBUG][{name}] {mask_name} 마스크가 원본 마스크와 겹치는 영역 있음")
            else:
                print(f"[DEBUG][{name}] {mask_name} 마스크가 원본 마스크와 겹치지 않음")
                
            accepted_masks.append(mask)
        else:
            m_sum = np.sum(mask)
            overlap = mask & orig_mask
            overlap_ratio = np.sum(overlap) / m_sum if m_sum > 0 else 0
            print(f"[DEBUG][{name}] Rejected {mask_name}, overlap_ratio={overlap_ratio:.2f}")

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

    print(f"[INFO][{name}] 감지된 마스크 수: {len(all_masks)}, 수락된 마스크 수: {len(accepted_masks)}")
    return accepted_masks

def process_batch(args):
    imgs, inp_dir, out_dir, cfg, ckpt, sam_type, dev, gs, steps = args
    debug_root = os.path.join(out_dir,'debug'); os.makedirs(debug_root,exist_ok=True)
    final_dir = os.path.join(out_dir,'final'); os.makedirs(final_dir,exist_ok=True)
    masks_dir = os.path.join(out_dir,'filled_masks'); os.makedirs(masks_dir,exist_ok=True)
    interim_dir = os.path.join(out_dir,'interim'); os.makedirs(interim_dir,exist_ok=True)

    device = f'cuda:{dev}' if torch.cuda.is_available() else 'cpu'
    model = load_model_from_config(OmegaConf.load(cfg), ckpt, device)
    sam   = get_sam_predictor(model_type=sam_type, device=device)

    for fn in tqdm(imgs, desc=f"GPU{dev}"):
        name, _ = os.path.splitext(fn)
        dr = os.path.join(debug_root, name)
        
        # 디버그 디렉토리 재생성
        if os.path.exists(dr):
            shutil.rmtree(dr)
        os.makedirs(dr, exist_ok=True)
            
        for sub in ['sam_masks', 'mask_input', 'comps', 'interim', 'over', 'masks']:
            os.makedirs(os.path.join(dr, sub), exist_ok=True)

        img = cv2.imread(os.path.join(inp_dir, fn))
        if img is None: 
            print(f"[ERROR] 이미지를 불러올 수 없습니다: {fn}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        white = np.all(img==255, axis=2)
        black = np.all(img==0, axis=2)
        orig_mask = white | black

        save_debug_image(orig_mask.astype(np.uint8)*255, os.path.join(dr, 'mask.png'))
        save_debug_image(img, os.path.join(dr, 'orig.png'))

        # 초기화 - 원본 이미지로 시작
        composite = img.copy()
        filled = np.zeros_like(orig_mask, dtype=bool)

        if orig_mask.sum() == 0:
            print(f"[INFO] 마스크가 없음: {fn}")
            Image.fromarray(img).save(os.path.join(final_dir, fn))
            continue

        sam.set_image(img)
        objs = find_objects_in_image(sam, img, orig_mask, name, dr)

        if len(objs) == 0:
            print(f"[WARNING] 적합한 객체가 없음: {fn}")
            Image.fromarray(img).save(os.path.join(final_dir, fn))
            continue

        # 모든 객체에 대한 완성 이미지를 수집
        completion_images = []
        
        for i, mask_bool in enumerate(objs, start=1):
            # debug: masked input to amodal
            masked_in = img.copy()
            for c in range(3):
                masked_in[:,:,c] = np.where(mask_bool, img[:,:,c], 0)
            save_debug_image(masked_in, os.path.join(dr, 'mask_input', f"{i}.png"))

            # overlay
            ov = visualize_mask_on_image(img, mask_bool)
            save_debug_image(ov, os.path.join(dr, 'over', f"{i}.png"))

            # mask_bool 마스크 저장
            save_debug_image(mask_bool.astype(np.uint8)*255, os.path.join(dr, 'masks',f"{i}.png"))

            # run_inference
            print(f"[DEBUG][{name}] run_inference with mask {i}, pixels={mask_bool.sum()}")
            comps = run_inference(img, mask_bool, model, gs, n_samples=1, ddim_steps=steps, device=device)
            print(f"[DEBUG][{name}] run_inference returned {len(comps)} samples")
            
            if not comps or len(comps) == 0: 
                print(f"[WARNING] 완성 결과 없음: {name}, 객체 {i}")
                continue
                
            comp = comps[0]
            # 데이터 타입 확인 및 변환
            if comp.dtype != np.uint8:
                print(f"[DEBUG][{name}] comp 데이터 타입 변환: {comp.dtype} -> uint8")
                if comp.max() <= 1.0:
                    comp = (comp * 255).astype(np.uint8)
                else:
                    comp = np.clip(comp, 0, 255).astype(np.uint8)
            
            save_debug_image(comp, os.path.join(dr, 'comps', f"{i}.png"))
            
            # 흰색이 아닌 픽셀 확인 (의미있는 생성인지 확인)
            non_white_mask = ~np.all(comp > 240, axis=2)
            non_white_count = np.sum(non_white_mask)
            print(f"[DEBUG][{name}] 흰색이 아닌 픽셀 수: {non_white_count}")
            
            # 흰색 픽셀만 있는 경우 스킵
            if non_white_count < 100:
                print(f"[INFO] 생성된 이미지가 대부분 흰색입니다: {name}, 객체 {i}")
                continue
            
            # 유효한 완성 이미지 저장
            completion_images.append(comp)
            
            # 중간 결과 저장 (개별 객체별 완성 이미지)
            interim_path = os.path.join(interim_dir, f"{name}_obj{i}.png")
            Image.fromarray(comp).save(interim_path)
            print(f"[INFO] 객체 {i} 완성 이미지 저장: {interim_path}")
        
        # 완성 이미지가 없는 경우
        if not completion_images:
            print(f"[WARNING] 유효한 완성 이미지가 없음: {name}")
            Image.fromarray(img).save(os.path.join(final_dir, fn))
            continue
        
        # ==================== 새로운 합성 로직 시작 ====================
        
        # 1) 마스크 이미지 읽기 & composite 초기화
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        mask[orig_mask] = 255  # 흰색/검은색 영역을 255로 설정
        inv_mask = cv2.bitwise_not(mask)
        
        # 합성 이미지 초기화 (원본 이미지로 시작)
        composite = img.copy()
        
        # 2) 모든 완성 이미지를 순차적으로 중첩
        for idx, comp_img in enumerate(completion_images, start=1):
            # 3) 완성 이미지에서 순수 흰/검은 픽셀 마스크
            white_px = np.all(comp_img == 255, axis=2).astype(np.uint8) * 255
            black_px = np.all(comp_img == 0, axis=2).astype(np.uint8) * 255
            no_fill = cv2.bitwise_or(white_px, black_px)    # 흰색 또는 검은색인 곳
            valid_px = cv2.bitwise_not(no_fill)             # 덧씌울 수 있는 픽셀
            
            # 4) 최종 적용 마스크 = 원래 mask 영역 ∧ valid_px
            apply_mask = cv2.bitwise_and(mask, valid_px)
            
            # 4-1) 디버그: 적용 마스크 저장
            save_debug_image(apply_mask, os.path.join(dr, 'masks', f"apply_mask_{idx}.png"))
            
            # 4-2) composite 갱신
            new_part = cv2.bitwise_and(comp_img, comp_img, mask=apply_mask)
            keep_part = cv2.bitwise_and(composite, composite, mask=cv2.bitwise_not(apply_mask))
            composite = cv2.add(new_part, keep_part)
            
            # 4-3) 중간 결과 저장 
            interim_path = os.path.join(dr, 'interim', f"composite_{idx}.png")
            save_debug_image(composite, interim_path)
            print(f"[INFO] 중간 합성 결과 저장: {interim_path}")
            
            # 4-4) 마스크 업데이트 (이미 채워진 부분은 제외)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(apply_mask))
        
        # 5) 최종 결과 저장
        save_debug_image(composite, os.path.join(dr, 'final_composite.png'))
        Image.fromarray(composite).save(os.path.join(final_dir, fn))
        
        # ==================== 새로운 합성 로직 끝 ====================
        
        # 채워진 마스크 비율 계산
        filled_mask = cv2.bitwise_xor(orig_mask.astype(np.uint8)*255, mask)
        fill_ratio = np.sum(filled_mask) / np.sum(orig_mask.astype(np.uint8)*255) * 100 if np.sum(orig_mask) > 0 else 0
        
        # 최종 채워진 마스크 저장
        save_debug_image(filled_mask, os.path.join(masks_dir, f"{name}_mask.png"))
        print(f"[INFO][GPU{dev}] 저장 완료: final/{fn}, 채움 비율={fill_ratio:.2f}%")
        
        # 메모리 정리
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    return True

def create_dataset(inp, out, cfg=None, ckpt=None, sam_model='vit_h', gs=2.0, steps=50, gpus=4, max_imgs=None):
    cfg = cfg or './configs/sd-finetune-pix2gestalt-c_concat-256.yaml'
    ckpt= ckpt or './ckpt/epoch=000005.ckpt'
    files = sorted([f for f in os.listdir(inp) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    if max_imgs: files=files[:max_imgs]
    print(f"총 {len(files)}개 이미지를 처리합니다.")
    
    ng=torch.cuda.device_count(); g = min(gpus,ng) if ng>0 else 1
    print(f"사용 가능한 GPU: {ng}개, 실제 사용: {g}개")
    
    bs=len(files)//g
    if bs == 0:
        bs = 1
    
    batches=[]
    for i in range(g):
        st,en=i*bs,min((i+1)*bs,len(files))
        if st>=len(files): break
        batches.append((files[st:en], inp, out, cfg, ckpt, sam_model, i, gs, steps))
    
    print(f"{g}개의 GPU로 분산 처리를 시작합니다.")
    
    if g == 1:
        process_batch(batches[0])
    else:
        with mp.Pool(g) as p:
            p.map(process_batch, batches)
    
    print(f"모든 처리 완료")

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input_dir',required=True, help='입력 이미지가 있는 디렉토리')
    p.add_argument('--output_dir',required=True, help='결과를 저장할 디렉토리')
    p.add_argument('--model_config', default=None, help='모델 설정 파일 경로')
    p.add_argument('--model_ckpt', default=None, help='모델 체크포인트 파일 경로')
    p.add_argument('--sam_model', default='vit_h', help='SAM 모델 타입 (vit_h, vit_l, vit_b)')
    p.add_argument('--guidance_scale', type=float, default=2.0, help='가이던스 스케일')
    p.add_argument('--ddim_steps', type=int, default=50, help='DDIM 스텝 수')
    p.add_argument('--num_gpus', type=int, default=4, help='사용할 GPU 수')
    p.add_argument('--max_images', type=int, default=None, help='처리할 최대 이미지 수')
    args=p.parse_args()
    
    create_dataset(
        args.input_dir, args.output_dir,
        args.model_config, args.model_ckpt,
        args.sam_model, args.guidance_scale,
        args.ddim_steps, args.num_gpus,
        args.max_images
    )

if __name__=='__main__':
    mp.set_start_method('spawn', force=True)
    main()