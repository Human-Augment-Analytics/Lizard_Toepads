#!/usr/bin/env python3
"""
Extract landmarks 1 & 2 (scale/ruler landmarks) from finger and toe TPS files
and consolidate them into separate scale TPS files.
"""
import os
from pathlib import Path
import argparse


def extract_scale_landmarks(tps_dir, output_dir, categories=['finger', 'toe']):
    """
    Extract landmarks 1 & 2 (scale/ruler) from TPS files and consolidate.
    
    Args:
        tps_dir: Directory containing individual TPS files
        output_dir: Output directory for consolidated scale TPS files
        categories: List of categories to process (default: ['finger', 'toe'])
    """
    tps_dir = Path(tps_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for category in categories:
        # Find all TPS files with this category suffix
        pattern = f'*_{category}.TPS'
        tps_files = sorted(tps_dir.glob(pattern))
        
        if not tps_files:
            print(f"No {category} TPS files found")
            continue
        
        output_file = output_dir / f'consolidated_{category}_scale.tps'
        
        print(f"\n{'='*60}")
        print(f"Extracting SCALE landmarks from {category.upper()} files")
        print(f"{'='*60}")
        print(f"Found {len(tps_files)} {category} TPS files")
        
        n_processed = 0
        n_skipped = 0
        
        with open(output_file, 'w') as outf:
            for tps_file in tps_files:
                try:
                    with open(tps_file, 'r') as inf:
                        lines = inf.readlines()
                    
                    # Process lines to extract only landmarks 1 & 2
                    processed_lines = []
                    i = 0
                    found_lm = False
                    found_scale = False
                    
                    while i < len(lines):
                        line = lines[i].strip()
                        
                        # Check if this is an LM line
                        if line.startswith('LM='):
                            try:
                                lm_num = int(line.split('=')[1])
                                
                                # We only want the first 2 landmarks (scale/ruler)
                                if lm_num >= 2:
                                    # Write LM=2 (only scale landmarks)
                                    processed_lines.append('LM=2\n')
                                    found_lm = True
                                    i += 1
                                    
                                    # Extract first 2 coordinate lines (landmarks 1 & 2)
                                    written = 0
                                    while i < len(lines) and written < 2:
                                        coord_line = lines[i].strip()
                                        # Check if it's a coordinate line
                                        if coord_line and (coord_line[0].isdigit() or coord_line.startswith('-')):
                                            processed_lines.append(lines[i])
                                            written += 1
                                            found_scale = True
                                        else:
                                            # If we hit a non-coordinate line, stop
                                            break
                                        i += 1
                                    
                                    # Skip remaining landmarks (3 onwards)
                                    skipped = 0
                                    while i < len(lines) and skipped < (lm_num - 2):
                                        coord_line = lines[i].strip()
                                        if coord_line and (coord_line[0].isdigit() or coord_line.startswith('-')):
                                            skipped += 1
                                        else:
                                            break
                                        i += 1
                                else:
                                    # If file has fewer than 2 landmarks, skip it
                                    print(f"  Warning: {tps_file.name} has only {lm_num} landmark(s), skipping")
                                    found_lm = False
                                    break
                            except:
                                # If parsing fails, skip this file
                                found_lm = False
                                break
                        
                        elif line.startswith('IMAGE='):
                            # Keep IMAGE line
                            processed_lines.append(lines[i])
                            i += 1
                        
                        elif line.startswith('ID=') or line.startswith('SCALE='):
                            # Keep ID and SCALE lines if present
                            processed_lines.append(lines[i])
                            i += 1
                        
                        elif not line or line.isspace():
                            # Skip empty lines
                            i += 1
                        
                        else:
                            # Other lines - skip after we've processed the landmarks
                            if found_lm:
                                i += 1
                            else:
                                # Still processing, copy the line
                                processed_lines.append(lines[i])
                                i += 1
                    
                    # Write processed content if we found scale landmarks
                    if found_scale and found_lm:
                        content = ''.join(processed_lines)
                        if content.strip():
                            outf.write(content)
                            # Ensure there's a newline between specimens
                            if not content.endswith('\n'):
                                outf.write('\n')
                            n_processed += 1
                        else:
                            print(f"  Skipping empty file: {tps_file.name}")
                            n_skipped += 1
                    else:
                        print(f"  Warning: {tps_file.name} - could not extract scale landmarks")
                        n_skipped += 1
                    
                except Exception as e:
                    print(f"  Error processing {tps_file.name}: {e}")
                    n_skipped += 1
        
        print(f"Created consolidated scale TPS file: {output_file}")
        print(f"  Successfully processed: {n_processed}")
        if n_skipped > 0:
            print(f"  Skipped: {n_skipped}")
        print(f"  Total {category} scale specimens: {n_processed}\n")


def verify_scale_file(file_path, category):
    """Verify the consolidated scale file."""
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
        
        # Verify all files have exactly 2 landmarks (scale/ruler)
        if lm_counts:
            if min(lm_counts.keys()) == 2 and max(lm_counts.keys()) == 2:
                print(f"    OK: All files have 2 landmarks (scale/ruler)")
            else:
                print(f"    WARNING: Some files may not have exactly 2 landmarks")


def main():
    parser = argparse.ArgumentParser(
        description='Extract scale/ruler landmarks (1 & 2) from finger and toe TPS files'
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
        help='Output directory for consolidated scale TPS files'
    )
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        choices=['finger', 'toe'],
        default=['finger', 'toe'],
        help='Categories to extract scale from (default: finger toe)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify consolidated files after creation'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("EXTRACTING SCALE/RULER LANDMARKS")
    print("="*60)
    print(f"Source directory: {args.tps_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Categories: {', '.join(args.categories)}")
    print(f"Extracting: Landmarks 1 & 2 (scale/ruler) from each file")
    
    # Extract scale landmarks
    extract_scale_landmarks(args.tps_dir, args.output_dir, args.categories)
    
    # Verify if requested
    if args.verify:
        print("\n" + "="*60)
        print("VERIFICATION")
        print("="*60)
        output_dir = Path(args.output_dir)
        for category in args.categories:
            output_file = output_dir / f'consolidated_{category}_scale.tps'
            print(f"\n{category.upper()} SCALE:")
            verify_scale_file(output_file, category)
    
    print("\n" + "="*60)
    print("Scale extraction complete!")
    print("="*60)


if __name__ == '__main__':
    main()

