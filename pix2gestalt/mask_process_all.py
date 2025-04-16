#!/usr/bin/env python3
"""
이 스크립트는 create_masked_images.py를 이용하여 여러 마스크 정책으로 이미지를 처리합니다.
"""
import os
import argparse
import subprocess
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="여러 마스크 정책으로 이미지를 처리")
    parser.add_argument("--input_dir", required=True, help="입력 이미지 디렉토리")
    parser.add_argument("--output_base_dir", required=True, help="출력 기본 디렉토리")
    
    # 모든 지원 정책 목록 (box_mask 추가)
    all_policies = [
        'random', 'batch_random', 'random_brush', 
        'seeded_random', 'seeded_brush', 'batch_seeded_random', 'replicated_random',
        'box_mask', 'box',  # box_mask 추가 (box도 별칭으로 추가)
        'grid', 'concentric', 'grid_probability', 'uniform_grid', 
        'box_grid', 'random_box_grid'
    ]
    
    parser.add_argument("--policies", nargs='+', 
                        default=['seeded_random', 'seeded_brush', 'box_mask', 'grid', 'concentric',
                                'grid_probability', 'uniform_grid', 'box_grid', 'random_box_grid'],
                        choices=all_policies,
                        help=f"사용할 마스크 정책 목록: {', '.join(all_policies)}")
    
    parser.add_argument("--mask_ratios", nargs='+', type=float, default=[0.3, 0.5, 0.7],
                        help="사용할 마스크 비율 목록")
    parser.add_argument("--num_images", type=int, default=None, 
                        help="각 정책당 처리할 이미지 수 (기본값: 모든 이미지)")
    parser.add_argument("--num_samples", type=int, default=1, 
                        help="각 이미지당 생성할 샘플 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--separate_folders", default=True,
                        help="마스크된 이미지와 마스크를 별도 폴더에 저장")
    
    args = parser.parse_args()
    
    # 출력 기본 디렉토리 생성
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # 각 마스크 비율에 대해 처리
    for mask_ratio in args.mask_ratios:
        # 마스크 비율에 따른 출력 디렉토리
        ratio_dir = os.path.join(args.output_base_dir, f"ratio_{mask_ratio:.1f}")
        os.makedirs(ratio_dir, exist_ok=True)
        
        # 마스크 정책 문자열 생성
        policies_str = ' '.join(args.policies)
        
        # create_masked_images.py 실행
        cmd = [
            "python", "create_masked_images.py",
            "--input_dir", args.input_dir,
            "--output_dir", ratio_dir,
            "--mask_policies"] + args.policies + [
            "--mask_ratio", str(mask_ratio),
            "--seed", str(args.seed),
            "--num_samples", str(args.num_samples)
        ]
        
        # 별도 폴더 옵션 추가
        if args.separate_folders:
            cmd.append("--separate_folders")
        
        # 이미지 수 제한이 설정된 경우 추가
        if args.num_images is not None:
            cmd += ["--num_images", str(args.num_images)]
        
        print(f"실행 명령어: {' '.join(cmd)}")
        subprocess.run(cmd)
    
    print("모든 마스크 비율 처리 완료!")

if __name__ == "__main__":
    main()