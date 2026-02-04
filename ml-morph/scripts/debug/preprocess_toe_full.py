#!/usr/bin/env python3
"""
Preprocess consolidated_toe.tps to create train/test XML files for full (non-cropped) images.

This script:
1. Reads consolidated_toe.tps
2. Splits images into train/ and test/ directories
3. Generates train.xml and test.xml for dlib training on full images
"""

import argparse
import os
import sys
from pathlib import Path
import utils

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess toe TPS file for full image training'
    )
    parser.add_argument(
        '-t', '--tps-file',
        type=str,
        default='consolidated_toe.tps',
        help='Path to consolidated TPS file (default: consolidated_toe.tps)'
    )
    parser.add_argument(
        '-i', '--image-dir',
        type=str,
        default='../data/miami_fall_24_jpgs',
        help='Directory containing original images (default: ../data/miami_fall_24_jpgs)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Ratio of images for training (default: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/test split (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.tps_file):
        print(f"Error: TPS file not found: {args.tps_file}")
        sys.exit(1)
    
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("Preprocessing Toe TPS File for Full Image Training")
    print("=" * 60)
    print(f"TPS file: {args.tps_file}")
    print(f"Image directory: {args.image_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Read TPS file
    print("\nReading TPS file...")
    dict_tps = utils.read_tps(args.tps_file)
    num_specimens = len(dict_tps['im'])
    print(f"Found {num_specimens} specimens in TPS file")
    
    # Split images into train/test
    print(f"\nSplitting images into train/test ({args.train_ratio:.0%}/{1-args.train_ratio:.0%})...")
    file_sizes = utils.split_train_test(args.image_dir)
    
    # Generate XML files
    print("\nGenerating train.xml...")
    utils.generate_dlib_xml(
        dict_tps,
        file_sizes['train'],
        folder='train',
        out_file='train.xml'
    )
    
    print("Generating test.xml...")
    utils.generate_dlib_xml(
        dict_tps,
        file_sizes['test'],
        folder='test',
        out_file='test.xml'
    )
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing complete!")
    print("=" * 60)
    print(f"Train images: {len(file_sizes['train'])}")
    print(f"Test images: {len(file_sizes['test'])}")
    print(f"\nGenerated files:")
    print(f"  - train.xml")
    print(f"  - test.xml")
    print(f"  - train/ (directory with training images)")
    print(f"  - test/ (directory with test images)")
    print("\nYou can now train the model using:")
    print("  python train_toe_full.py")

if __name__ == '__main__':
    main()

