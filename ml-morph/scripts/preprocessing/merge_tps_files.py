#!/usr/bin/env python3
"""
Merge individual TPS files into consolidated TPS files for ml-morph training.
Each specimen in the TPS directory gets merged into a single master TPS file.
"""
import os
import argparse
from pathlib import Path


def merge_tps_files(tps_dir, output_file, suffix='_finger'):
    """
    Merge all TPS files with a given suffix into a single consolidated TPS file.

    Args:
        tps_dir: Directory containing individual TPS files
        output_file: Output path for consolidated TPS file
        suffix: File suffix to filter (e.g., '_finger', '_toe')
    """
    tps_files = sorted([f for f in os.listdir(tps_dir) if f.endswith(f'{suffix}.TPS')])

    print(f"Found {len(tps_files)} {suffix} TPS files")

    with open(output_file, 'w') as outf:
        for tps_file in tps_files:
            tps_path = os.path.join(tps_dir, tps_file)
            with open(tps_path, 'r') as inf:
                content = inf.read()
                outf.write(content)
                # Ensure there's a newline between specimens
                if not content.endswith('\n'):
                    outf.write('\n')

    print(f"Created consolidated TPS file: {output_file}")
    print(f"Total specimens: {len(tps_files)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge individual TPS files into consolidated files')
    parser.add_argument('--tps-dir', required=True, help='Directory containing individual TPS files')
    parser.add_argument('--output-dir', default='.', help='Output directory for consolidated files')
    parser.add_argument('--types', nargs='+', default=['finger', 'toe'],
                        help='Types to merge (default: finger toe)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for tps_type in args.types:
        output_file = os.path.join(args.output_dir, f'consolidated_{tps_type}.tps')
        merge_tps_files(args.tps_dir, output_file, suffix=f'_{tps_type}')
