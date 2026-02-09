import os
import glob
import re

def clean_filenames(base_dir):
    """
    Renames files in base_dir/images and base_dir/labels.
    Expected format: ID_jpg.rf.HASH.jpg -> ID.jpg
                     ID_jpg.rf.HASH.txt -> ID.txt
    """
    
    for subdir in ['images', 'labels']:
        dir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue
            
        print(f"Processing {dir_path}...")
        files = os.listdir(dir_path)
        count = 0
        
        for filename in files:
            # Match pattern: digits + _jpg + .rf. + hash + extension
            # Example: 1001_jpg.rf.28c1ac4244714cf4ad37f5dcf9ae04b4.jpg
            match = re.match(r"(\d+)_jpg\.rf\.[a-zA-Z0-9]+\.(jpg|txt)", filename)
            
            if match:
                file_id = match.group(1)
                ext = match.group(2)
                new_filename = f"{file_id}.{ext}"
                
                old_path = os.path.join(dir_path, filename)
                new_path = os.path.join(dir_path, new_filename)
                
                # Check if target already exists to avoid overwriting
                if os.path.exists(new_path):
                    print(f"Warning: {new_filename} already exists. Skipping {filename}.")
                    continue
                
                os.rename(old_path, new_path)
                count += 1
                # print(f"Renamed: {filename} -> {new_filename}")
        
        print(f"Renamed {count} files in {subdir}.")

if __name__ == "__main__":
    # Adjust path as needed. Assuming script is run from project root or scripts/preprocessing
    # The absolute path is safer.
    dataset_dir = "/home/hice1/jzhuang48/Lizard_Toepads/data/upper_dataset_roboflow/train"
    clean_filenames(dataset_dir)
