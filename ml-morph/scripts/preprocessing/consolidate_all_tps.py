#!/usr/bin/env python3
"""
Consolidate TPS files by category:
- id: Consolidate all _id.TPS files (already have 2 landmarks)
- finger: Consolidate all _finger.TPS files (keep all landmarks including 1 & 2)
- toe: Consolidate all _toe.TPS files (keep all landmarks including 1 & 2)
"""
import os
from pathlib import Path
import argparse


def consolidate_tps_files(tps_dir, output_dir, categories=None, remove_landmarks_1_2=False):
    """
    Consolidate TPS files by category.
    
    Args:
        tps_dir: Directory containing individual TPS files
        output_dir: Output directory for consolidated TPS files
        categories: List of categories to process (default: ['finger', 'toe'])
        remove_landmarks_1_2: If True, remove landmarks 1 & 2 during consolidation (for scale/ruler)
    """
    tps_dir = Path(tps_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if categories is None:
        categories = ['finger', 'toe']  # Skip ID by default
    
    for category in categories:
        # Find all TPS files with this category suffix
        pattern = f'*_{category}.TPS'
        tps_files = sorted(tps_dir.glob(pattern))
        
        if not tps_files:
            print(f"No {category} TPS files found")
            continue
        
        output_file = output_dir / f'consolidated_{category}.tps'
        
        print(f"\n{'='*60}")
        print(f"Processing {category.upper()} files")
        print(f"{'='*60}")
        print(f"Found {len(tps_files)} {category} TPS files")
        
        n_processed = 0
        n_skipped = 0
        
        with open(output_file, 'w') as outf:
            for tps_file in tps_files:
                try:
                    with open(tps_file, 'r') as inf:
                        lines = inf.readlines()
                    
                    # Process lines
                    processed_lines = []
                    i = 0
                    
                    while i < len(lines):
                        line = lines[i].strip()
                        
                        # Check if this is an LM line
                        if line.startswith('LM='):
                            try:
                                lm_num = int(line.split('=')[1])
                                
                                # If removing landmarks 1 & 2 for finger/toe
                                if remove_landmarks_1_2 and category in ['finger', 'toe'] and lm_num >= 2:
                                    # Write updated LM count (remove 2 landmarks)
                                    new_lm_num = lm_num - 2
                                    processed_lines.append(f'LM={new_lm_num}\n')
                                    i += 1
                                    
                                    # Skip first 2 coordinate lines (landmarks 1 & 2)
                                    skipped = 0
                                    while i < len(lines) and skipped < 2:
                                        coord_line = lines[i].strip()
                                        # Check if it's a coordinate line
                                        if coord_line and (coord_line[0].isdigit() or coord_line.startswith('-')):
                                            skipped += 1
                                        i += 1
                                    
                                    # Write remaining landmarks (landmarks 3 onwards, now numbered 1 onwards)
                                    written = 0
                                    while i < len(lines) and written < new_lm_num:
                                        coord_line = lines[i].strip()
                                        # Check if it's a coordinate line
                                        if coord_line and (coord_line[0].isdigit() or coord_line.startswith('-')):
                                            processed_lines.append(lines[i])
                                            written += 1
                                        else:
                                            # If we hit a non-coordinate line before getting all landmarks, stop
                                            break
                                        i += 1
                                else:
                                    # Write LM line and all landmarks as-is (for ID or if not removing)
                                    processed_lines.append(lines[i])
                                    i += 1
                                    
                                    # Write all landmark coordinates
                                    written = 0
                                    while i < len(lines) and written < lm_num:
                                        coord_line = lines[i].strip()
                                        if coord_line and (coord_line[0].isdigit() or coord_line.startswith('-')):
                                            processed_lines.append(lines[i])
                                            written += 1
                                        else:
                                            break
                                        i += 1
                            except:
                                # If parsing fails, just copy the line
                                processed_lines.append(lines[i])
                                i += 1
                        
                        elif line.startswith('IMAGE=') or line.startswith('ID=') or line.startswith('SCALE='):
                            # Keep metadata lines
                            processed_lines.append(lines[i])
                            i += 1
                        
                        elif not line or line.isspace():
                            # Skip empty lines
                            i += 1
                        
                        else:
                            # Other lines (shouldn't happen in well-formed TPS, but handle it)
                            processed_lines.append(lines[i])
                            i += 1
                    
                    # Write processed content
                    content = ''.join(processed_lines)
                    if not content.strip():
                        print(f"  Skipping empty file: {tps_file.name}")
                        n_skipped += 1
                        continue
                    
                    outf.write(content)
                    # Ensure there's a newline between specimens
                    if not content.endswith('\n'):
                        outf.write('\n')
                    
                    n_processed += 1
                    
                except Exception as e:
                    print(f"  Error processing {tps_file.name}: {e}")
                    n_skipped += 1
        
        print(f"Created consolidated TPS file: {output_file}")
        print(f"  Successfully processed: {n_processed}")
        if n_skipped > 0:
            print(f"  Skipped: {n_skipped}")
        print(f"  Total {category} specimens: {n_processed}\n")


def verify_consolidated_file(file_path, category):
    """Verify the consolidated file and show summary statistics."""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"  Warning: Consolidated file not found: {file_path}")
        return
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    n_specimens = content.count('IMAGE=')
    lm_lines = [l for l in lines if l.startswith('LM=')]
    
    if lm_lines:
        # Count unique landmark counts
        lm_counts = {}
        i = 0
        while i < len(lines):
            if lines[i].startswith('LM='):
                try:
                    count = int(lines[i].split('=')[1].strip())
                    lm_counts[count] = lm_counts.get(count, 0) + 1
                except:
                    pass
            i += 1
        
        print(f"  Verification:")
        print(f"    Total specimens: {n_specimens}")
        print(f"    Landmark counts: {dict(sorted(lm_counts.items()))}")
        
        if category in ['finger', 'toe']:
            # Check if landmarks 1 & 2 were removed (should have LM = original - 2)
            if lm_counts:
                # Typical finger/toe files have 11 landmarks, after removing 1&2 should have 9
                if 9 in lm_counts:
                    print(f"    OK: Landmarks 1 & 2 removed (scale/ruler), now have {min(lm_counts.keys())}-{max(lm_counts.keys())} landmarks")
                elif min(lm_counts.keys()) >= 2:
                    print(f"    OK: Files have {min(lm_counts.keys())}-{max(lm_counts.keys())} landmarks")
                else:
                    print(f"    WARNING: Some files may have fewer than expected landmarks")


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate TPS files by category (id, finger, toe)'
    )
    parser.add_argument(
        '--tps-dir',
        type=str,
        default='ml-morph/tps_files',
        help='Directory containing individual TPS files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='ml-morph',
        help='Output directory for consolidated TPS files'
    )
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        choices=['id', 'finger', 'toe'],
        default=['finger', 'toe'],
        help='Categories to consolidate (default: finger toe, excluding id)'
    )
    parser.add_argument(
        '--remove-scale-landmarks',
        action='store_true',
        default=True,
        help='Remove landmarks 1 & 2 (scale/ruler) from finger/toe files (default: True)'
    )
    parser.add_argument(
        '--keep-scale-landmarks',
        action='store_false',
        dest='remove_scale_landmarks',
        help='Keep landmarks 1 & 2 in finger/toe files'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify consolidated files after creation'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("TPS FILE CONSOLIDATION")
    print("="*60)
    print(f"Source directory: {args.tps_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Categories: {', '.join(args.categories)}")
    if args.remove_scale_landmarks:
        print(f"Removing landmarks 1 & 2 (scale/ruler) from finger/toe files")
    else:
        print(f"Keeping all landmarks including 1 & 2")
    
    # Consolidate files
    consolidate_tps_files(args.tps_dir, args.output_dir, args.categories, 
                         remove_landmarks_1_2=args.remove_scale_landmarks)
    
    # Verify if requested
    if args.verify:
        print("\n" + "="*60)
        print("VERIFICATION")
        print("="*60)
        output_dir = Path(args.output_dir)
        for category in args.categories:
            output_file = output_dir / f'consolidated_{category}.tps'
            print(f"\n{category.upper()}:")
            verify_consolidated_file(output_file, category)
    
    print("\n" + "="*60)
    print("Consolidation complete!")
    print("="*60)


if __name__ == '__main__':
    main()

