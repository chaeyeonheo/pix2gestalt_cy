'''
#CLI
# 기본 예시 ─ 필수 인자만 지정
python amodal_enhanced_dataset.py  --input_dir ~/volum1/cy/latent-code-inpainting_new/data/data_large/test_256 --max_images 10  --output_dir ~/volum1/cy/pix2gestalt/results/amodal_dataset1  --num_gpus 6         

'''


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

    composite_dir = os.path.join(out_dir, "composites")
    intermediate_root = os.path.join(out_dir, "intermediate")
    debug_root = os.path.join(out_dir, "debug")
    masks_dir = os.path.join(out_dir, "filled_masks")

    os.makedirs(composite_dir, exist_ok=True)
    os.makedirs(intermediate_root, exist_ok=True)
    os.makedirs(debug_root, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    device = f'cuda:{dev}' if torch.cuda.is_available() else 'cpu'
    model = load_model_from_config(OmegaConf.load(cfg), ckpt, device)
    sam = get_sam_predictor(model_type=sam_type, device=device)

    for fn in tqdm(imgs, desc=f"GPU{dev}"):
        name, _ = os.path.splitext(fn)

        # 디렉토리 세팅
        debug_dir = os.path.join(debug_root, name)
        interim_dir = os.path.join(intermediate_root, name)
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(interim_dir, exist_ok=True)
        for sub in ['sam_masks', 'mask_input', 'comps', 'interim', 'over', 'masks']:
            os.makedirs(os.path.join(debug_dir, sub), exist_ok=True)

        img_path = os.path.join(inp_dir, fn)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] 이미지를 불러올 수 없습니다: {fn}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        white = np.all(img == 255, axis=2)
        black = np.all(img == 0, axis=2)
        orig_mask = white | black

        save_debug_image(orig_mask.astype(np.uint8) * 255, os.path.join(debug_dir, 'mask.png'))
        save_debug_image(img, os.path.join(debug_dir, 'orig.png'))

        if orig_mask.sum() == 0:
            print(f"[INFO] 마스크가 없음: {fn}")
            Image.fromarray(img).save(os.path.join(composite_dir, f"{name}.png"))
            continue

        sam.set_image(img)
        objs = find_objects_in_image(sam, img, orig_mask, name, debug_dir)

        if len(objs) == 0:
            print(f"[WARNING] 적합한 객체가 없음: {fn}")
            Image.fromarray(img).save(os.path.join(composite_dir, f"{name}.png"))
            continue

        # 합성용 마스크 및 이미지 초기화
        orig_mask_uint8 = orig_mask.astype(np.uint8) * 255
        composite = img.copy()

        for idx, mask_bool in enumerate(objs, start=1):
            visible_mask = (mask_bool.astype(np.uint8) * 255)

            # 마스크 입력 저장
            masked_in = img.copy()
            for c in range(3):
                masked_in[:, :, c] = np.where(mask_bool, img[:, :, c], 0)
            save_debug_image(masked_in, os.path.join(debug_dir, 'mask_input', f"{idx}.png"))
            save_debug_image(visible_mask, os.path.join(debug_dir, 'masks', f"{idx}.png"))
            save_debug_image(visualize_mask_on_image(img, mask_bool), os.path.join(debug_dir, 'over', f"{idx}.png"))

            results = run_inference(
                input_image=img,
                visible_mask=visible_mask,
                model=model,
                guidance_scale=gs,
                n_samples=1,
                ddim_steps=steps,
                device=device
            )

            if not results:
                print(f"[WARNING] 객체 {idx}에 대한 생성 실패")
                continue

            comp_img = results[0]
            if comp_img.dtype != np.uint8:
                comp_img = np.clip(comp_img * 255, 0, 255).astype(np.uint8)

            save_debug_image(comp_img, os.path.join(debug_dir, 'comps', f"{idx}.png"))

            white_px = cv2.inRange(comp_img, np.array([255, 255, 255]), np.array([255, 255, 255]))
            black_px = cv2.inRange(comp_img, np.array([0, 0, 0]), np.array([0, 0, 0]))
            no_fill = cv2.bitwise_or(white_px, black_px)
            valid_px = cv2.bitwise_not(no_fill)

            apply_mask = cv2.bitwise_and(orig_mask_uint8, valid_px)

            if apply_mask.sum() < 10:
                print(f"[INFO] 유효 픽셀이 너무 적어 스킵: {name}, 객체 {idx}")
                continue

            new_part = cv2.bitwise_and(comp_img, comp_img, mask=apply_mask)
            keep_part = cv2.bitwise_and(composite, composite, mask=cv2.bitwise_not(apply_mask))
            composite = cv2.add(new_part, keep_part)

            save_debug_image(apply_mask, os.path.join(debug_dir, 'masks', f"apply_mask_{idx}.png"))
            save_debug_image(composite, os.path.join(debug_dir, 'interim', f"composite_{idx}.png"))
            save_debug_image(composite, os.path.join(interim_dir, f"composite_{idx}.png"))
            print(f"[INFO] 객체 {idx} 적용 완료")

            orig_mask_uint8 = cv2.bitwise_and(orig_mask_uint8, cv2.bitwise_not(apply_mask))

        # 최종 composite 결과 저장
        final_path = os.path.join(composite_dir, f"{name}.png")
        save_debug_image(composite, os.path.join(debug_dir, 'final_composite.png'))
        Image.fromarray(composite).save(final_path)

        filled_mask = cv2.bitwise_xor(orig_mask.astype(np.uint8) * 255, orig_mask_uint8)
        fill_ratio = np.sum(filled_mask) / np.sum(orig_mask.astype(np.uint8) * 255) * 100 if np.sum(orig_mask) > 0 else 0
        save_debug_image(filled_mask, os.path.join(masks_dir, f"{name}_mask.png"))
        print(f"[INFO][GPU{dev}] 저장 완료: {final_path}, 채움 비율={fill_ratio:.2f}%")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return True

def create_dataset(inp, out, cfg=None, ckpt=None, sam_model='vit_h', gs=2.0, steps=50, gpus=6, max_imgs=None):
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
    p.add_argument('--num_gpus', type=int, default=6, help='사용할 GPU 수')
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