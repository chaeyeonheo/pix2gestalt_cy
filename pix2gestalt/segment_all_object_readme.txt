결과 구조
이제 결과 디렉토리 구조는 다음과 같습니다:
output_dir/
├── all_amodal/                     # 통합 시각화 기본 디렉토리
│   ├── combined/                  # 통합 이미지 디렉토리
│   │   ├── image1.png            # 이미지1의 모든 객체 amodal 결과 통합
│   │   ├── image2.png
│   │   └── ...
│   └── combined_masks/            # 통합 마스크 디렉토리
│       ├── image1.png            # 이미지1의 통합 마스크
│       ├── image2.png
│       └── ...
├── image1/
│   ├── image1_original.png         # 원본 이미지
│   ├── image1_all_objects.png      # 모든 객체 시각화
│   ├── object_000/                 # 첫 번째 객체
│   │   ├── mask.png                # 객체 마스크
│   │   ├── masked_input.png        # 마스킹된 입력 이미지
│   │   └── amodal_result_0.png     # amodal completion 결과
│   ├── object_001/                 # 두 번째 객체
│   └── ...
├── image2/
└── ...
사용 방법
코드 사용 방법은 이전과 동일합니다:
python segment_all_objects.py --input_dir /path/to/images --output_dir /path/to/results --gpu_ids 0,1,2,3,4,5,6,7
이렇게 하면 output_dir/all_amodal/combined 폴더에 모든 객체의 amodal completion 결과가 합쳐진 이미지가 저장되고, output_dir/all_amodal/combined_masks 폴더에 해당 마스크가 저장됩니다.
이미지 이름을 유지하여 관리가 용이하고, 디렉토리 구조를 통해 이미지와 마스크를 명확히 구분할 수 있습니다.