import numpy as np
import torch
import torch.nn as nn
import math
import random
import cv2
from PIL import Image, ImageDraw

# I/O functions
def write_images(path, image, n_row=1):
    image = ((image + 1) * 255 / 2).astype(np.uint8)
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite('{}'.format(str(path)), np.squeeze(image))

# Mask functions
def RandomBrush(
    max_tries,
    s,
    min_num_vertex = 4,
    max_num_vertex = 18,
    mean_angle = 2*math.pi / 5,
    angle_range = 2*math.pi / 15,
    min_width = 12,
    max_width = 48):
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask

def RandomMask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)
        MultiFill(int(5 * coef), s // 2) # 3
        MultiFill(int(3 * coef), s)      # 2
        mask = np.logical_and(mask, 1 - RandomBrush(int(9 * coef), s))  # hole denoted as 0, reserved as 1 # 4
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)

def BatchRandomMask(batch_size, s, hole_range=[0, 1]):
    return np.stack([RandomMask(s, hole_range=hole_range) for _ in range(batch_size)], axis=0)


# 시드 고정 버전의 RandomBrush 함수 추가
def SeededRandomBrush(
    max_tries,
    s,
    seed=42,
    min_num_vertex = 4,
    max_num_vertex = 18,
    mean_angle = 2*math.pi / 5,
    angle_range = 2*math.pi / 15,
    min_width = 12,
    max_width = 48):
    """
    시드를 고정하여 항상 동일한 브러시 패턴을 생성하는 함수
    
    Args:
        max_tries: 최대 시도 횟수
        s: 마스크 크기
        seed: 난수 생성기 시드 (기본값: 42)
        기타 매개변수: 브러시 스트로크 형태 제어
    
    Returns:
        생성된 브러시 마스크
    """
    # 현재 numpy 난수 상태 저장
    np_state = np.random.get_state()
    random_state = random.getstate()
    
    # 시드 설정
    np.random.seed(seed)
    random.seed(seed)
    
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    
    # 원래 numpy 난수 상태 복원
    np.random.set_state(np_state)
    random.setstate(random_state)
    
    return mask


# 시드 고정 버전의 RandomMask 함수 추가
def SeededRandomMask(s, hole_range=[0,1], seed=42):
    """
    시드를 고정하여 항상 동일한 랜덤 마스크를 생성하는 함수
    
    Args:
        s: 마스크 크기
        hole_range: 마스크 구멍 비율 범위
        seed: 난수 생성기 시드 (기본값: 42)
    
    Returns:
        생성된 마스크
    """
    # 현재 numpy 난수 상태 저장
    np_state = np.random.get_state()
    random_state = random.getstate()
    
    # 시드 설정
    np.random.seed(seed)
    random.seed(seed)
    
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)
        MultiFill(int(5 * coef), s // 2) # 3
        MultiFill(int(3 * coef), s)      # 2
        mask = np.logical_and(mask, 1 - SeededRandomBrush(int(9 * coef), s, seed=seed+1))  # hole denoted as 0, reserved as 1 # 4
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        
        # 원래 numpy 난수 상태 복원
        np.random.set_state(np_state)
        random.setstate(random_state)
        
        return mask[np.newaxis, ...].astype(np.float32)


# 동일한 랜덤 마스크를 배치 크기만큼 복제하는 함수
def ReplicatedRandomMask(batch_size, s, hole_range=[0, 1], seed=42):
    """
    단일 랜덤 마스크를 생성한 후 배치 크기만큼 복제하는 함수
    
    Args:
        batch_size: 배치 크기
        s: 마스크 크기
        hole_range: 마스크 구멍 비율 범위
        seed: 난수 생성기 시드 (기본값: 42)
    
    Returns:
        복제된 마스크 세트
    """
    # 단일 마스크 생성
    single_mask = SeededRandomMask(s, hole_range=hole_range, seed=seed)
    # 배치 크기만큼 복제
    return np.repeat(single_mask, batch_size, axis=0)


# 각기 다른 랜덤 마스크를 배치 크기만큼 생성하되, 시드를 고정하여 재현 가능하게 하는 함수
def BatchSeededRandomMask(batch_size, s, hole_range=[0, 1], base_seed=42):
    """
    배치 크기만큼 서로 다른 마스크를 생성하되, 시드를 고정하여 항상 동일한 마스크 세트를 생성하는 함수
    
    Args:
        batch_size: 배치 크기
        s: 마스크 크기
        hole_range: 마스크 구멍 비율 범위
        base_seed: 기본 시드 (각 마스크마다 다른 시드 생성에 사용)
    
    Returns:
        생성된 마스크 배치
    """
    masks = []
    for i in range(batch_size):
        # 각 마스크마다 base_seed에 인덱스를 더한 값을 시드로 사용
        mask = SeededRandomMask(s, hole_range=hole_range, seed=base_seed+i)
        masks.append(mask)
    return np.concatenate(masks, axis=0)


def grid_probability_mask(shape, device, grid_size=16, p_range=(0.2, 0.8)):
    """
    이미지를 grid로 나누고 각 grid마다 다른 확률로 마스킹
    
    Args:
        shape: 입력 텐서의 shape (batch_size, channels, height, width)
        device: 텐서가 있는 디바이스
        grid_size: grid의 크기 (grid_size x grid_size)
        p_range: 마스킹 확률의 범위 (min_p, max_p)
    """
    batch_size, _, height, width = shape
    
    # grid 크기 계산
    h_grids = height // grid_size
    w_grids = width // grid_size
    
    # 각 grid마다 랜덤한 확률 생성 (p_range 범위 내에서)
    grid_probs = torch.rand(batch_size, 1, h_grids, w_grids, device=device)
    grid_probs = grid_probs * (p_range[1] - p_range[0]) + p_range[0]
    
    # grid 확률을 원래 이미지 크기로 업샘플링
    grid_probs = torch.nn.functional.interpolate(
        grid_probs, 
        size=(height, width), 
        mode='nearest'
    )
    
    # 확률에 따라 마스크 생성
    mask = torch.bernoulli(1 - grid_probs)  # 1 - p로 반전 (1이 보존, 0이 마스크)
    mask = mask.repeat(1, shape[1], 1, 1)  # 채널 차원 복제
    
    return mask

def random_box_grid_probability_mask(shape, device, grid_size=16, p_range=(0.2, 0.8)):
    """
    이미지를 grid로 나누고 각 grid마다 다른 크기의 박스로 마스킹
    
    Args:
        shape: 입력 텐서의 shape (batch_size, channels, height, width)
        device: 텐서가 있는 디바이스
        grid_size: grid의 크기 (grid_size x grid_size)
        p_range: 마스킹 비율의 범위 (min_p, max_p)
    
    Returns:
        mask: 마스킹된 텐서
        grid_ps: 각 grid의 실제 p값을 담은 텐서 (batch_size, h_grids, w_grids)
    """
    batch_size, _, height, width = shape
    
    # grid 개수 계산
    h_grids = height // grid_size
    w_grids = width // grid_size
    
    # 각 grid마다 랜덤한 p값 생성
    grid_ps = torch.rand(batch_size, h_grids, w_grids, device=device)
    grid_ps = grid_ps * (p_range[1] - p_range[0]) + p_range[0]
    
    # 결과 마스크 초기화 (1이 보존, 0이 마스크)
    mask = torch.ones(shape, device=device)
    
    for b in range(batch_size):
        for h in range(h_grids):
            for w in range(w_grids):
                # 현재 grid의 p값으로 박스 크기 계산
                current_p = grid_ps[b, h, w].item()
                box_size = int(np.sqrt(current_p) * grid_size)
                box_size = max(1, min(box_size, grid_size))
                
                # 각 grid의 시작점
                h_start = h * grid_size
                w_start = w * grid_size
                
                # 박스의 랜덤 시작 위치 계산
                h_offset = torch.randint(0, grid_size - box_size + 1, (1,)).item()
                w_offset = torch.randint(0, grid_size - box_size + 1, (1,)).item()
                
                # 박스 위치
                h_box_start = h_start + h_offset
                w_box_start = w_start + w_offset
                
                # 박스 마스킹
                mask[b, :, h_box_start:h_box_start+box_size, 
                     w_box_start:w_box_start+box_size] = 0
    
    return mask, grid_ps


def uniform_grid_probability_mask(shape, device, grid_size=16, p=0.5):
    """
    이미지를 grid로 나누고 각 grid 내에서 동일한 확률 p로 마스킹
    
    Args:
        shape: 입력 텐서의 shape (batch_size, channels, height, width)
        device: 텐서가 있는 디바이스
        grid_size: grid의 크기 (grid_size x grid_size)
        p: 각 grid 내에서의 마스킹 확률
    """
    batch_size, _, height, width = shape
    
    # grid 크기 계산
    h_grids = height // grid_size
    w_grids = width // grid_size
    
    # 모든 grid에 동일한 확률 p 적용
    grid_probs = torch.ones(batch_size, 1, h_grids, w_grids, device=device) * p
    
    # grid 확률을 원래 이미지 크기로 업샘플링
    grid_probs = torch.nn.functional.interpolate(
        grid_probs, 
        size=(height, width), 
        mode='nearest'
    )
    
    # 확률에 따라 마스크 생성
    mask = torch.bernoulli(1 - grid_probs)  # 1 - p로 반전 (1이 보존, 0이 마스크)
    mask = mask.repeat(1, shape[1], 1, 1)  # 채널 차원 복제
    
    return mask

def box_grid_probability_mask(shape, device, grid_size=16, p=0.5):
    """
    이미지를 grid로 나누고 각 grid 내에서 p 크기의 박스로 마스킹
    
    Args:
        shape: 입력 텐서의 shape (batch_size, channels, height, width)
        device: 텐서가 있는 디바이스
        grid_size: grid의 크기 (grid_size x grid_size)
        p: 각 grid 내에서 박스가 차지할 비율 (0~1)
    """
    batch_size, _, height, width = shape
    
    # grid 개수 계산
    h_grids = height // grid_size
    w_grids = width // grid_size
    
    # 박스 크기 계산 (p에 비례하여)
    box_size = int(np.sqrt(p) * grid_size)
    box_size = max(1, min(box_size, grid_size))  # 박스 크기를 1과 grid_size 사이로 제한
    
    # 결과 마스크 초기화 (1이 보존, 0이 마스크)
    mask = torch.ones(shape, device=device)
    
    for b in range(batch_size):
        for h in range(h_grids):
            for w in range(w_grids):
                # 각 grid의 시작점
                h_start = h * grid_size
                w_start = w * grid_size
                
                # 박스의 랜덤 시작 위치 계산 (grid 내에서)
                h_offset = torch.randint(0, grid_size - box_size + 1, (1,)).item()
                w_offset = torch.randint(0, grid_size - box_size + 1, (1,)).item()
                
                # 박스 위치
                h_box_start = h_start + h_offset
                w_box_start = w_start + w_offset
                
                # 박스 마스킹 (모든 채널에 대해)
                mask[b, :, h_box_start:h_box_start+box_size, 
                     w_box_start:w_box_start+box_size] = 0
    
    return mask

def grid_mask(shape, device, grid_size=16, p=0.5):
    """
    이미지를 grid로 나누고 각 grid를 몇개 정도 masking할지

    """
    batch_size, _, height, width = shape
    
    # grid 크기 계산
    h_grids = height // grid_size
    w_grids = width // grid_size
    
    # 각 grid에 대해 동일한 확률로 마스크 생성
    grid_mask = torch.bernoulli(
        (1-p) * torch.ones(batch_size, 1, h_grids, w_grids, device=device)
    )
    
    # grid 마스크를 원래 이미지 크기로 업샘플링
    mask = torch.nn.functional.interpolate(
        grid_mask, 
        size=(height, width), 
        mode='nearest'
    )
    
    # 채널 차원 복제
    mask = mask.repeat(1, shape[1], 1, 1)
    
    return mask


def concentric_box_mask(shape, device, num_boxes=3, box_gap=20, box_width=10):
    """
    여러 개의 띠 형태의 중첩된 박스 마스크 생성 (각 박스 사이 빈 공간 유지)

    Args:
        shape: 입력 텐서의 shape (batch_size, channels, height, width)
        device: 텐서가 있는 디바이스
        num_boxes: 생성할 박스의 수
        box_gap: 박스 간 간격 (박스가 커지는 크기)
        box_width: 각 띠의 두께
    """
    batch_size, _, height, width = shape
    mask = torch.ones(batch_size, 1, height, width, device=device)  # 기본적으로 흰색(1)

    # 중심점
    center_h, center_w = height // 2, width // 2
    
    for i in range(num_boxes):
        # 박스 크기 (점점 커지면서 빈 공간 유지)
        size = (i + 1) * box_gap * 2
        
        if size >= min(height, width):
            break  # 박스가 화면을 벗어나면 중지
            
        # 바깥 박스 경계
        outer_h = center_h - size // 2
        outer_w = center_w - size // 2
        end_h = center_h + size // 2
        end_w = center_w + size // 2
        
        # 안쪽 박스 경계 (띠 형태 유지)
        inner_h = outer_h + box_width
        inner_w = outer_w + box_width
        inner_end_h = end_h - box_width
        inner_end_w = end_w - box_width

        # 여러 개의 박스를 유지하면서 업데이트
        mask[:, :, outer_h:end_h, outer_w:outer_w+box_width] = 0  # 왼쪽 띠
        mask[:, :, outer_h:end_h, end_w-box_width:end_w] = 0      # 오른쪽 띠
        mask[:, :, outer_h:outer_h+box_width, outer_w:end_w] = 0  # 위쪽 띠
        mask[:, :, end_h-box_width:end_h, outer_w:end_w] = 0      # 아래쪽 띠

    # 채널 차원 복제 (입력 이미지 채널 수에 맞추기)
    mask = mask.repeat(1, shape[1], 1, 1)
    
    return mask