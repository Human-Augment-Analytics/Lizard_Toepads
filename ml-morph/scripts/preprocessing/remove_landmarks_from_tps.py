#!/usr/bin/env python3
"""
Remove specific landmarks from TPS file.
This script removes landmarks 1 and 2 from each entry.
"""
import sys
from pathlib import Path


def remove_landmarks(tps_file_path: str, output_path: str, landmarks_to_remove: list = [1, 2]):
    """
    Remove specified landmarks from TPS file.
    
    Args:
        tps_file_path: Path to input TPS file
        output_path: Path to output TPS file
        landmarks_to_remove: List of 1-based landmark indices to remove (default: [1, 2])
    """
    tps_file_path = Path(tps_file_path)
    output_path = Path(output_path)
    
    if not tps_file_path.exists():
        print(f"Error: TPS file not found: {tps_file_path}")
        return False
    
    print(f"Reading TPS file: {tps_file_path}")
    
    with open(tps_file_path, 'r') as f:
        lines = f.readlines()
    
    output_lines = []
    i = 0
    n_processed = 0
    landmarks_removed = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this is an LM line
        if line.startswith('LM='):
            # Get number of landmarks
            num_landmarks = int(line.split('=')[1])
            output_lines.append(line + '\n')
            i += 1
            
            # Read all landmarks for this entry
            landmark_lines = []
            for j in range(num_landmarks):
                if i < len(lines):
                    landmark_lines.append(lines[i].strip())
                    i += 1
            
            # Remove specified landmarks (1-based indexing)
            # Convert to 0-based for list indexing
            indices_to_remove = [idx - 1 for idx in landmarks_to_remove if 1 <= idx <= num_landmarks]
            indices_to_remove.sort(reverse=True)  # Sort in reverse to remove from end first
            
            for idx in indices_to_remove:
                if 0 <= idx < len(landmark_lines):
                    landmark_lines.pop(idx)
                    landmarks_removed += 1
            
            # Update LM count
            new_num_landmarks = len(landmark_lines)
            output_lines[-1] = f'LM={new_num_landmarks}\n'
            
            # Write remaining landmarks
            for lm_line in landmark_lines:
                output_lines.append(lm_line + '\n')
            
            n_processed += 1
            
        elif line.startswith('IMAGE='):
            # Keep IMAGE line as-is
            output_lines.append(line + '\n')
            i += 1
        elif line.startswith('SCALE='):
            # Keep SCALE line as-is (if present)
            output_lines.append(line + '\n')
            i += 1
        elif not line or line.isspace():
            # Skip empty lines
            i += 1
        else:
            # This shouldn't happen in well-formed TPS files, but handle it
            i += 1
    
    # Write output file
    print(f"Writing output to: {output_path}")
    with open(output_path, 'w') as f:
        f.writelines(output_lines)
    
    print(f"\nProcessing complete:")
    print(f"  - Processed {n_processed} images")
    print(f"  - Removed {landmarks_removed} landmark entries total")
    print(f"  - Removed landmarks: {landmarks_to_remove}")
    print(f"  - Output saved to: {output_path}")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Remove specified landmarks from TPS file'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='ml-morph/consolidated_finger.tps',
        help='Input TPS file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output TPS file path (default: overwrites input with .bak backup)'
    )
    parser.add_argument(
        '--landmarks',
        type=int,
        nargs='+',
        default=[1, 2],
        help='1-based landmark indices to remove (default: 1 2)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Create backup and overwrite original
        backup_path = input_path.with_suffix(input_path.suffix + '.bak')
        print(f"Creating backup: {backup_path}")
        import shutil
        shutil.copy2(input_path, backup_path)
        output_path = input_path
    
    success = remove_landmarks(
        str(input_path),
        str(output_path),
        landmarks_to_remove=args.landmarks
    )
    
    if not success:
        sys.exit(1)

