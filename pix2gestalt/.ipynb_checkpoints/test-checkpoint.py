import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import sys
import os

# 필요한 모듈 임포트
from inference import load_model_from_config, get_sam_predictor, run_sam, run_inference

# 설정 파일 로드
config_path = "configs/pix2gestalt_config.yaml"
config = OmegaConf.load(config_path)

# 모델 로드
checkpoint_path = "ckpt/pix2gestalt_model.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_from_config(config, checkpoint_path, device)

# 입력 이미지 로드
input_image_path = "/volum1/cy/pix2gestalt/results/dataset8/debug/Places365_test_00000723/ori.png"
input_image = np.array(Image.open(input_image_path).resize((256, 256)))

# SAM 예측기 초기화 및 이미지 설정
predictor = get_sam_predictor(model_type='vit_h', device=device, image=input_image)

# 포인트 좌표로 객체 선택 (예: 이미지 중앙)
selected_points = [([input_image.shape[1]//2, input_image.shape[0]//2], 1)]  # (좌표, 레이블)

# SAM으로 마스크 생성
visible_mask, _ = run_sam(predictor, selected_points)

# 모델 추론 실행
results = run_inference(
    input_image=input_image,
    visible_mask=visible_mask,
    model=model,
    guidance_scale=2.0,
    n_samples=1,
    ddim_steps=50,
    device=device
)

# 결과 저장
for i, result in enumerate(results):
    Image.fromarray(result).save(f"result_{i}.png")