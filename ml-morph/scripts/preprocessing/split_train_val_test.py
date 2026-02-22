#!/usr/bin/env python3
"""
Split data into train/validation/test sets for better model evaluation.
Validation set is used for hyperparameter tuning, test set for final evaluation.
"""
import os
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom


def parse_xml(xml_file):
    """Parse dlib XML and extract image entries."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    images = []

    for images_elem in root.findall('images'):
        for image_elem in images_elem.findall('image'):
            images.append(image_elem)

    return root, images


def create_xml_subset(root_template, image_elements, output_file):
    """Create a new XML file with subset of images."""
    root = ET.Element('dataset')
    root.append(ET.Element('name'))
    root.append(ET.Element('comment'))

    images_elem = ET.Element('images')
    root.append(images_elem)

    for img_elem in image_elements:
        images_elem.append(img_elem)

    # Pretty print
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(output_file, "w") as f:
        f.write(xmlstr)


def split_data(images_dir, split_ratio=(0.7, 0.15, 0.15), seed=845):
    """
    Split images into train/val/test sets.

    Args:
        images_dir: Directory containing all images
        split_ratio: Tuple of (train, val, test) ratios (should sum to 1.0)
        seed: Random seed for reproducibility

    Returns:
        dict: Dictionary with 'train', 'val', 'test' lists of filenames
    """
    assert abs(sum(split_ratio) - 1.0) < 0.001, "Split ratios must sum to 1.0"

    # Get all image files
    all_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
    all_files.sort()

    # Shuffle with seed
    random.seed(seed)
    random.shuffle(all_files)

    # Calculate split points
    n = len(all_files)
    train_end = int(n * split_ratio[0])
    val_end = train_end + int(n * split_ratio[1])

    splits = {
        'train': all_files[:train_end],
        'val': all_files[train_end:val_end],
        'test': all_files[val_end:]
    }

    print(f"Total images: {n}")
    print(f"Train: {len(splits['train'])} ({len(splits['train'])/n*100:.1f}%)")
    print(f"Val: {len(splits['val'])} ({len(splits['val'])/n*100:.1f}%)")
    print(f"Test: {len(splits['test'])} ({len(splits['test'])/n*100:.1f}%)")

    return splits


def reorganize_files(source_dir, splits, base_output_dir='.'):
    """
    Copy images to train/val/test directories.
    """
    for split_name, files in splits.items():
        split_dir = os.path.join(base_output_dir, split_name)

        # Create or clean directory
        if os.path.exists(split_dir):
            print(f"Cleaning {split_dir}/")
            shutil.rmtree(split_dir)
        os.makedirs(split_dir)

        # Copy files
        print(f"Copying {len(files)} files to {split_dir}/")
        for fname in files:
            src = os.path.join(source_dir, fname)
            dst = os.path.join(split_dir, fname)
            shutil.copy2(src, dst)


def create_xml_files(original_train_xml, original_test_xml, splits, base_output_dir='.'):
    """
    Create train.xml, val.xml, and test.xml based on splits.
    """
    # Parse both original XMLs
    print("Parsing original XML files...")
    train_root, train_images = parse_xml(original_train_xml)
    test_root, test_images = parse_xml(original_test_xml)

    # Combine all images
    all_images = train_images + test_images

    # Create mapping from filename to image element
    image_map = {}
    for img_elem in all_images:
        file_attr = img_elem.get('file')
        # Extract just the filename (remove path)
        filename = os.path.basename(file_attr)
        image_map[filename] = img_elem

    # Create XML for each split
    for split_name, files in splits.items():
        print(f"Creating {split_name}.xml with {len(files)} images...")

        # Get image elements for this split
        split_images = []
        missing_count = 0
        for fname in files:
            if fname in image_map:
                img_elem = image_map[fname]
                # Update the file path to point to correct directory
                new_elem = ET.fromstring(ET.tostring(img_elem))
                new_elem.set('file', f'{split_name}/{fname}')
                split_images.append(new_elem)
            else:
                missing_count += 1
                print(f"  Warning: No annotations found for {fname}")

        if missing_count > 0:
            print(f"  Skipped {missing_count} images without annotations")

        # Create XML file
        output_file = os.path.join(base_output_dir, f'{split_name}.xml')
        create_xml_subset(train_root, split_images, output_file)
        print(f"  Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Split data into train/val/test sets')
    parser.add_argument('--images-dir', default='images', help='Directory with all images')
    parser.add_argument('--train-xml', default='train.xml', help='Original training XML')
    parser.add_argument('--test-xml', default='test.xml', help='Original test XML')
    parser.add_argument('--output-dir', default='.', help='Base output directory')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=845, help='Random seed')
    parser.add_argument('--backup', action='store_true', help='Backup existing train/test dirs')

    args = parser.parse_args()

    split_ratio = (args.train_ratio, args.val_ratio, args.test_ratio)

    print("="*60)
    print("SPLITTING DATA INTO TRAIN/VAL/TEST")
    print("="*60)
    print(f"Split ratio: {split_ratio[0]:.0%} / {split_ratio[1]:.0%} / {split_ratio[2]:.0%}")
    print(f"Random seed: {args.seed}")
    print()

    # Backup existing directories if requested
    if args.backup:
        for dirname in ['train', 'test', 'val']:
            dirpath = os.path.join(args.output_dir, dirname)
            if os.path.exists(dirpath):
                backup_path = f"{dirpath}_backup"
                print(f"Backing up {dirpath} to {backup_path}")
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
                shutil.move(dirpath, backup_path)

    # Split the data
    splits = split_data(args.images_dir, split_ratio, args.seed)

    # Reorganize files
    print("\n" + "="*60)
    print("ORGANIZING FILES")
    print("="*60)
    reorganize_files(args.images_dir, splits, args.output_dir)

    # Create XML files
    print("\n" + "="*60)
    print("CREATING XML FILES")
    print("="*60)
    create_xml_files(args.train_xml, args.test_xml, splits, args.output_dir)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("You can now train with:")
    print(f"  python shape_trainer.py -d train.xml -t val.xml -o model_name")
    print("\nFor final evaluation, use:")
    print(f"  python shape_tester.py test.xml model_name.dat")


if __name__ == '__main__':
    main()
