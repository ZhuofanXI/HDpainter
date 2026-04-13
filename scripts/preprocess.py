"""
Offline preprocessing script for HDpainter.

Steps:
  1. check_and_annotate  — label each raw tile with center_nuclei_count (in-place).
  2. build_filtered_dataset — filter, convert sparse->dense, apply SVD scale, save.

Usage:
  uv run python scripts/preprocess.py \
      --src_dir  /path/to/data \
      --dst_dir  /path/to/output \
      --min_nuc  10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import build_filtered_dataset, check_and_annotate


def parse_args():
    p = argparse.ArgumentParser(description="HDpainter offline data preprocessing")
    p.add_argument("--src_dir",    required=True, help="Parent dir containing per-dataset subdirs (e.g. SVD_CESC/)")
    p.add_argument("--dst_dir",    required=True, help="Output dir for processed tiles")
    p.add_argument("--min_nuc",    type=int, default=5, help="Min unique nuclei in centre crop to keep a tile")
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--overlap",    type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"=== Step 1: check_and_annotate ===")
    print(f"src_dir={args.src_dir}, patch_size={args.patch_size}, overlap={args.overlap}")
    check_and_annotate(args.src_dir, patch_size=args.patch_size, overlap=args.overlap)

    print(f"\n=== Step 2: build_filtered_dataset ===")
    print(f"src_dir={args.src_dir}, dst_dir={args.dst_dir}, min_nuc={args.min_nuc}")
    build_filtered_dataset(args.src_dir, args.dst_dir, min_nuc=args.min_nuc)

    print("\nPreprocessing complete.")
