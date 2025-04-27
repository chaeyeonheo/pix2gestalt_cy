import numpy as np
import torch
import torch.nn as nn
import math
import cv2
from PIL import Image, ImageDraw

# --- Box Mask ---
def box_mask(shape, device, p=0.5, det=False):
    nb, _, r, _ = shape
    mr = int(p * r)
    mask = np.ones([nb, 1, r, r]).astype(np.int32)
    for i in range(nb):
        if det:
            h = w = (r - mr) // 2
        else:
            h, w = np.random.randint(0, r - mr, 2)
        mask[i, :, h:h+mr, w:w+mr] = 0
    return torch.from_numpy(mask).to(device)


def center_box_mask(shape, device, p=0.5):
    nb, _, r, _ = shape
    mr = int(p * r)
    start = (r - mr) // 2
    end = start + mr
    mask = np.ones([nb, 1, r, r], dtype=np.int32)
    mask[:, :, start:end, start:end] = 0
    return torch.from_numpy(mask).to(device)

# --- Scatter Mask ---
def scatter_mask(shape, device, p=0.5):
    p = (1 - p)
    mask = torch.bernoulli(p * torch.ones(shape, device=device)).round().long()
    return mask

# --- Mixed Mask ---
def mixed_mask(shape, device, p=0.5, scatter_prob=0.5):
    if np.random.rand() < scatter_prob:
        return scatter_mask(shape, device, p)
    else:
        return box_mask(shape, device, p)

# --- Grid Probability Mask ---
def grid_probability_mask(shape, device, grid_size=16, p_range=(0.2, 0.8)):
    B, C, H, W = shape
    h_grids, w_grids = H // grid_size, W // grid_size
    probs = torch.rand(B, 1, h_grids, w_grids, device=device) * (p_range[1] - p_range[0]) + p_range[0]
    upsampled = torch.nn.functional.interpolate(probs, size=(H, W), mode='nearest')
    return torch.bernoulli(1 - upsampled).repeat(1, C, 1, 1)

# --- Uniform Grid Probability Mask ---
def uniform_grid_probability_mask(shape, device, grid_size=16, p=0.5):
    B, C, H, W = shape
    h_grids, w_grids = H // grid_size, W // grid_size
    probs = torch.ones(B, 1, h_grids, w_grids, device=device) * p
    upsampled = torch.nn.functional.interpolate(probs, size=(H, W), mode='nearest')
    return torch.bernoulli(1 - upsampled).repeat(1, C, 1, 1)

# --- Box Grid Probability Mask ---
def box_grid_probability_mask(shape, device, grid_size=16, p=0.5):
    B, C, H, W = shape
    h_grids, w_grids = H // grid_size, W // grid_size
    box_size = max(1, min(int(np.sqrt(p) * grid_size), grid_size))
    mask = torch.ones(shape, device=device)
    for b in range(B):
        for h in range(h_grids):
            for w in range(w_grids):
                hs, ws = h * grid_size, w * grid_size
                ho = torch.randint(0, grid_size - box_size + 1, (1,)).item()
                wo = torch.randint(0, grid_size - box_size + 1, (1,)).item()
                mask[b, :, hs+ho:hs+ho+box_size, ws+wo:ws+wo+box_size] = 0
    return mask

# --- Random Mask + Brush ---
def RandomBrush(max_tries, s, min_num_vertex=4, max_num_vertex=18,
                mean_angle=2*math.pi/5, angle_range=2*math.pi/15,
                min_width=12, max_width=48):
    H, W = s, s
    avg_radius = math.sqrt(H*H + W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = [2*math.pi - np.random.uniform(angle_min, angle_max) if i % 2 == 0 else np.random.uniform(angle_min, angle_max) for i in range(num_vertex)]
        vertex = [(np.random.randint(0, W), np.random.randint(0, H))]
        for angle in angles:
            r = np.clip(np.random.normal(loc=avg_radius, scale=avg_radius//2), 0, 2*avg_radius)
            x, y = vertex[-1][0] + r * math.cos(angle), vertex[-1][1] + r * math.sin(angle)
            vertex.append((int(np.clip(x, 0, W)), int(np.clip(y, 0, H))))
        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0]-width//2, v[1]-width//2, v[0]+width//2, v[1]+width//2), fill=1)
    return np.array(mask, dtype=np.uint8)

def RandomMask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            x, y = np.random.randint(-w//2, s - w + w//2), np.random.randint(-h//2, s - h + h//2)
            mask[max(y,0):min(y+h,s), max(x,0):min(x+w,s)] = 0
        for _ in range(np.random.randint(int(5*coef))): Fill(s//2)
        for _ in range(np.random.randint(int(3*coef))): Fill(s)
        brush_mask = RandomBrush(int(9*coef), s)
        final = np.logical_and(mask, 1 - brush_mask)
        hole_ratio = 1 - np.mean(final)
        if hole_range[0] <= hole_ratio <= hole_range[1]:
            return torch.from_numpy(final[np.newaxis, ...].astype(np.float32))

# --- Multiple Ratio Mask ---
def multiple_ratio_mask(shape, device, policies_with_ratios):
    """
    여러 마스킹 정책을 비율에 따라 무작위로 적용합니다.
    
    Args:
        shape: 이미지 형태 (B, C, H, W)
        device: 텐서 장치
        policies_with_ratios: 정책 설정 목록 [(정책 이름, 파라미터, 비율)]
                             예: [('box', 0.5, 0.4), ('center_box', 0.8, 0.3), ('random', 0.3, 0.3)]
    Returns:
        선택된 마스킹 정책으로 생성된 마스크
    """
    # 무작위 값을 사용하여 정책 선택
    r = np.random.random()
    cumulative = 0.0
    
    for policy_name, param, ratio in policies_with_ratios:
        cumulative += ratio
        if r < cumulative:
            # 마스킹 정책 선택 및 적용
            if policy_name == 'random':
                # RandomMask는 특별한 처리가 필요합니다
                rand_mask = RandomMask(shape[-1], hole_range=[0, float(param)])
                # 텐서로 변환하고 배치 및 채널 차원 추가
                rand_mask = rand_mask.repeat(shape[0], shape[1], 1, 1).to(device)
                return rand_mask
            elif policy_name == 'box':
                return box_mask(shape, device, p=float(param))
            elif policy_name == 'center_box':
                return center_box_mask(shape, device, p=float(param))
            elif policy_name == 'scatter':
                return scatter_mask(shape, device, p=float(param))
            elif policy_name == 'grid':
                return grid_probability_mask(shape, device, p=float(param))
            elif policy_name == 'uniform_grid':
                return uniform_grid_probability_mask(shape, device, p=float(param))
            elif policy_name == 'box_grid':
                return box_grid_probability_mask(shape, device, p=float(param))
            elif policy_name == 'mixed':
                return mixed_mask(shape, device, p=float(param))
    
    # 기본값 (이 코드가 실행되는 경우는 없어야 함)
    return box_mask(shape, device, p=0.5)