# 이미지 마스킹 도구

다양한 마스킹 정책을 적용하여 이미지에 마스크를 생성하는 도구입니다. 이 도구는 이미지 인페인팅 연구나 아모달 세그멘테이션(amodal segmentation) 실험에 유용합니다.

## 파일 구조

- `mask_utils.py`: 다양한 마스크 생성 함수들이 포함된 유틸리티 모듈
- `create_masked_images.py`: 이미지에 여러 마스크 정책을 적용하고 결과를 저장하는 메인 스크립트
- `process_all.py`: 여러 마스크 비율과 정책으로 배치 처리를 실행하는 래퍼 스크립트

## 필요 라이브러리 설치

```bash
pip install numpy torch pillow opencv-python tqdm
```

## 지원되는 마스크 정책

1. **random**: 시드 없는 랜덤 마스크
2. **batch_random**: 배치 형태의 랜덤 마스크
3. **random_brush**: 시드 없는 붓 형태 마스크
4. **seeded_random**: 시드 고정 랜덤 마스크
5. **seeded_brush**: 시드 고정 붓 형태 마스크
6. **batch_seeded_random**: 배치 형태의 시드 고정 랜덤 마스크
7. **replicated_random**: 복제된 시드 고정 랜덤 마스크
8. **box_mask** / **box**: 박스 형태 마스크 (중앙에 사각형으로 마스킹)
9. **grid**: 그리드 기반 마스크
10. **concentric**: 동심원 형태의 띠 마스크
11. **grid_probability**: 그리드마다 다른 확률로 마스킹
12. **uniform_grid**: 균일한 확률의 그리드 마스크
13. **box_grid**: 그리드마다 박스 형태로 마스킹
14. **random_box_grid**: 그리드마다 다른 크기의 박스로 마스킹

## 사용법

### 1. 단일 마스크 비율로 처리하기

```bash
python create_masked_images.py --input_dir /path/to/images --output_dir /path/to/output --mask_policies seeded_random box_mask grid --mask_ratio 0.5 --num_images 100
```

**인자 설명:**
- `--input_dir`: 입력 이미지 디렉토리
- `--output_dir`: 출력 디렉토리
- `--mask_policies`: 사용할 마스크 정책 목록
- `--mask_ratio`: 마스크 비율 (0-1)
- `--num_images`: 처리할 이미지 수 (생략 시 모든 이미지)
- `--num_samples`: 각 이미지당 생성할 샘플 수 (기본값: 1)
- `--seed`: 랜덤 시드 (기본값: 42)

### 2. 여러 마스크 비율과 정책으로 처리하기

```bash
python process_all.py --input_dir /path/to/images --output_base_dir /path/to/output --policies seeded_random box_mask grid --mask_ratios 0.3 0.5 0.7 --num_images 100
```

**인자 설명:**
- `--input_dir`: 입력 이미지 디렉토리
- `--output_base_dir`: 출력 기본 디렉토리
- `--policies`: 사용할 마스크 정책 목록
- `--mask_ratios`: 사용할 마스크 비율 목록
- `--num_images`: 처리할 이미지 수 (생략 시 모든 이미지)
- `--num_samples`: 각 이미지당 생성할 샘플 수 (기본값: 1)
- `--seed`: 랜덤 시드 (기본값: 42)

## 출력 결과

각 정책마다 별도의 폴더가 생성되며, 그 안에 다음 파일들이 저장됩니다:
- `{image_name}.png`: 마스크가 적용된 이미지
- `{image_name}_mask.png`: 적용된 마스크 (흰색(255)은 보존 영역, 검은색(0)은 마스킹 영역)

여러 샘플을 생성하는 경우:
- `{image_name}_sample{N}.png`: N번째 샘플 이미지
- `{image_name}_sample{N}_mask.png`: N번째 샘플의 마스크

## 예제

1. 모든 이미지에 대해 seeded_random, box_mask와 grid 정책을 적용:
```bash
python create_masked_images.py --input_dir ./images --output_dir ./outputs --mask_policies seeded_random box_mask grid
```

2. 100개 이미지에 대해 세 가지 마스크 비율로 여러 정책 적용:
```bash
python process_all.py --input_dir ./images --output_base_dir ./outputs --policies seeded_random box_mask grid concentric --mask_ratios 0.3 0.5 0.7 --num_images 100
```

3. 각 이미지마다 세 개의 다른 샘플 생성:
```bash
python create_masked_images.py --input_dir ./images --output_dir ./outputs --mask_policies box_mask --num_samples 3
```