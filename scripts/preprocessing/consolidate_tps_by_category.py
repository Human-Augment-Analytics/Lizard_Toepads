#!/usr/bin/env python3
"""
Consolidate TPS files by category (id, toe, finger) into single files.
"""
import os
from pathlib import Path


def consolidate_tps_files(tps_dir, output_dir):
    """
    Consolidate TPS files into three categories: id, toe, and finger.
    
    Args:
        tps_dir: Directory containing individual TPS files
        output_dir: Output directory for consolidated TPS files
    """
    tps_dir = Path(tps_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    categories = ['id', 'toe', 'finger']
    
    for category in categories:
        # Find all TPS files with this category suffix
        pattern = f'*_{category}.TPS'
        tps_files = sorted(tps_dir.glob(pattern))
        
        if not tps_files:
            print(f"No {category} TPS files found")
            continue
        
        output_file = output_dir / f'consolidated_{category}.tps'
        
        print(f"Found {len(tps_files)} {category} TPS files")
        
        with open(output_file, 'w') as outf:
            for tps_file in tps_files:
                with open(tps_file, 'r') as inf:
                    content = inf.read()
                    outf.write(content)
                    # Ensure there's a newline between specimens
                    if not content.endswith('\n'):
                        outf.write('\n')
        
        print(f"Created consolidated TPS file: {output_file}")
        print(f"Total {category} specimens: {len(tps_files)}\n")


if __name__ == '__main__':
    # Default paths
    tps_dir = Path(__file__).parent.parent.parent / 'data' / 'tps_files'
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'tps_files'
    
    consolidate_tps_files(tps_dir, output_dir)

