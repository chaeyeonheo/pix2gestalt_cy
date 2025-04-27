# Transformer 기반 Amodal Fusion Network

이 프로젝트는 Transformer 아키텍처를 기반으로 여러 개의 Amodal Completion 결과를 학습 기반으로 중첩하여 더 나은 Image Inpainting 입력을 만들기 위한 네트워크를 구현합니다.

## 개요

다중 객체가 있는 장면에서 마스크된 영역이 여러 amodal completion 결과에 의해 중첩되는 경우, Transformer는 전역 정보를 활용하여 어떤 객체의 완성 결과를 각 픽셀에 사용할지 결정합니다. 이 접근법은 객체 간의 관계와 장면 구조를 더 효과적으로 이해할 수 있습니다.

## 주요 컴포넌트

1. **AmodalFusionTransformer**: Transformer 인코더-디코더 기반 네트워크로, 부분 이미지와 여러 completion 이미지를 입력받아 각 completion에 대한 픽셀별 가중치 맵을 생성합니다.
2. **2D 위치 인코딩**: 공간적 정보를 Transformer에 제공하기 위한 2D 위치 인코딩 구현.
3. **분산 훈련 지원**: 여러 GPU에서 효율적으로 모델을 훈련하기 위한 DistributedDataParallel 구현.
4. **추론 스크립트**: 학습된 모델을 사용하여 새로운 이미지에 대한 가중치 맵을 생성하고 fusion 결과를 시각화합니다.

## 아키텍처 특징

- **컨볼루션 인코더**: 이미지를 다운샘플링하고 초기 특징을 추출하는 컨볼루션 레이어
- **Transformer 인코더**: 이미지 픽셀 간의 전역 관계를 모델링하는 멀티헤드 어텐션 기반 인코더
- **Transformer 디코더**: 전체 이미지 정보를 쿼리하고 처리하는 디코더
- **업샘플링 디코더**: 처리된 특징을 원래 해상도로 복원하는 컨볼루션 레이어
- **가중치 헤드**: 각 completion에 대한 픽셀별 가중치를 생성하는 최종 레이어

## 설치 및 의존성

```bash
pip install torch torchvision tqdm pillow matplotlib numpy opencv-python lpips pyyaml
```

## 사용 방법

### 1. 데이터셋 준비

prepare_dataset.py를 사용하여 훈련용 데이터셋을 준비합니다:

```bash
python prepare_dataset.py create \
  --input_dir /path/to/amodal_results \
  --output_dir /path/to/weightnet_dataset \
  --n_max_completions 5 \
  --img_size 256 \
  --resize \
  --split
```

### 2. 모델 훈련 (6 GPU 사용)

설정 파일을 사용한 분산 훈련:

```bash
python amodal_fusion_transformer.py \
  --config transformer_train_config.yaml
```

또는 직접 인수 지정:

```bash
python amodal_fusion_transformer.py \
  --data_dir /path/to/weightnet_dataset \
  --train_list /path/to/weightnet_dataset/train_list.txt \
  --val_list /path/to/weightnet_dataset/val_list.txt \
  --img_size 256 \
  --n_max_completions 5 \
  --batch_size 8 \
  --epochs 50 \
  --lr 0.0001 \
  --gpus 6
```

### 3. 추론 실행

학습된 모델을 사용하여 완성 이미지 융합:

```bash
python amodal_fusion_transformer_inference.py \
  --data_dir /path/to/test_data \
  --checkpoint checkpoints/transformer_run1/best_model.pth \
  --output_dir /path/to/output \
  --n_max_completions 5 \
  --img_size 256
```

## 파일 구조 개요

- `amodal_fusion_transformer.py`: Transformer 모델 정의 및 훈련 코드 (분산 훈련 지원)
- `amodal_fusion_transformer_inference.py`: 학습된 모델을 사용한 추론 코드
- `prepare_dataset.py`: 데이터셋 준비 및 분할 도구
- `transformer_train_config.yaml`: 훈련 설정 예제

## Transformer의 장점

1. **전역 문맥 이해**: Transformer의 self-attention 메커니즘은 이미지의 모든 픽셀 간 관계를 모델링하여 전역적 문맥을 이해할 수 있습니다. 이를 통해 중첩된 객체들 간의 관계와 장면의 구조를 더 효과적으로 파악합니다.

2. **occlusion 관계 학습**: 단순히 픽셀별 가중치가 아닌, 객체 간의 전반적인 관계와 깊이 순서를 암묵적으로 학습할 수 있습니다.

3. **적응형 문맥 처리**: 어텐션 메커니즘은 각 픽셀에 대해 가장 관련성 높은 문맥 정보를 동적으로 집중할 수 있어, 복잡한 장면에서도 정확한 융합이 가능합니다.

Transformer 기반 접근법은 특히 중첩이 많거나 복잡한 장면에서, 그리고 객체 간의 관계가 중요한 상황에서 더 강력한 성능을 보여줍니다.

## 훈련 절차

1. 모델은 부분 이미지, 마스크, 여러 amodal completion 이미지를 입력으로 받습니다.
2. Transformer 인코더-디코더 구조는 전체 이미지를 처리하고 각 픽셀에 대한 가중치 맵을 생성합니다.
3. 이 가중치를 사용하여 completion 이미지들을 융합합니다.
4. 원본 이미지와 비교하여 L1 손실과 LPIPS 지각적 손실을 계산합니다.
5. 분산 훈련을 통해 여러 GPU에서 효율적으로 모델을 최적화합니다.

이 접근 방식을 통해 inpainting 모델에 더 일관된 입력을 제공하여 최종적으로 더 나은 결과를 얻을 수 있습니다.