파일 구조 및 사용법
project_root/
├── amodal_fusion_dataset.py                    # 데이터셋 클래스
├── amodal_fusion_transformer_folder.py         # 훈련 코드
├── amodal_fusion_transformer_inference_folder.py  # 추론 코드
└── transformer_train_config_folder.yaml        # 설정 파일
훈련하기
이미지별 폴더 구조를 사용한 훈련:
bashpython amodal_fusion_transformer_folder.py \
  --train_dir /path/to/train_dataset \
  --val_dir /path/to/val_dataset \
  --gpus 6 \
  --batch_size 8 \
  --epochs 50 \
  --checkpoint_dir checkpoints/transformer_run1
또는 설정 파일 사용:
bashpython amodal_fusion_transformer_folder.py --config transformer_train_config_folder.yaml
추론하기
학습된 모델을 사용해 추론:
bashpython amodal_fusion_transformer_inference_folder.py \
  --input_dir /path/to/test_dataset \
  --checkpoint checkpoints/transformer_run1/best_model.pth \
  --output_dir /path/to/outputs
코드 설명
amodal_fusion_dataset.py
이미지별 폴더 구조를 지원하는 데이터셋 클래스를 제공합니다. 자동으로 각 이미지 폴더를 스캔하여 필요한 파일이 있는지 확인하고, 유효한 이미지 ID만 처리합니다.
amodal_fusion_transformer_folder.py
Transformer 기반 AmodalFusionNet의 훈련 코드입니다. 단일 GPU 및 다중 GPU(최대 6개) 훈련을 지원합니다. 주요 기능:

전역 문맥을 이해하는 Transformer 아키텍처
L1 및 LPIPS 지각적 손실을 통한 훈련
분산 데이터 병렬(DDP) 훈련 지원
자동 체크포인트 저장 및 최고 모델 추적

amodal_fusion_transformer_inference_folder.py
학습된 모델을 사용하여 새로운 이미지에 대한 추론을 수행합니다. 입력 폴더에서 유효한 이미지 ID를 자동으로 찾아 처리합니다.
transformer_train_config_folder.yaml
훈련 설정 파일 예시입니다. 다음과 같은 설정을 조정할 수 있습니다:

데이터셋 경로
모델 아키텍처 매개변수 (d_model, nhead 등)
훈련 하이퍼파라미터 (배치 크기, 학습률 등)
GPU 수
출력 디렉토리

예상 결과
훈련이 완료되면 다음과 같은 결과물이 생성됩니다:

체크포인트: 지정된 checkpoint_dir에 모델 가중치가 저장됩니다.
시각화: 훈련 및 검증 중 샘플 시각화가 vis_dir에 저장됩니다.
로그: 훈련 손실 및 메트릭이 log_dir에 저장됩니다 (TensorBoard로 확인 가능).

추론을 실행하면 다음과 같은 결과물이 생성됩니다:

융합된 이미지: 여러 amodal completion이 최적으로 중첩된 최종 결과
가중치 맵: 각 completion에 대한 픽셀별 가중치 시각화
시각화 그리드: 입력, 중간 단계, 최종 결과를 모두 보여주는 종합 이미지

환경 및 의존성
필요한 라이브러리:

PyTorch (1.8 이상)
torchvision
LPIPS
matplotlib
numpy
tqdm
PyYAML

성능 최적화 팁

배치 크기: GPU 메모리에 맞게 조정하세요. 고해상도 이미지는 작은 배치 크기가 필요할 수 있습니다.
다중 GPU: 6개 GPU를 사용할 경우 훈련 속도가 크게 향상됩니다. 분산 학습 설정(--gpus 6)을 활용하세요.
그래디언트 클리핑: 훈련이 불안정하다면 --grad_clip 값을 조정해보세요.
학습률: 기본값 0.0001이 대부분의 경우 잘 작동하지만, 더 나은 결과를 위해 0.00005나 0.0002를 시도해볼 수 있습니다.