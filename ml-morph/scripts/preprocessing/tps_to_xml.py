#!/usr/bin/env python3
"""
Simple TPS to XML converter for PyTorch keypoint training.
No dlib dependency required.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET
from xml.dom import minidom

from PIL import Image

# Allow very large images (normalized images can be ~10k x 10k)
Image.MAX_IMAGE_PIXELS = None


def parse_tps_file(tps_path: Path) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """Parse TPS file and return list of (image_name, landmarks)."""
    annotations = []

    with open(tps_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for LM= line
        if line.startswith('LM='):
            num_landmarks = int(line.split('=')[1])
            landmarks = []

            # Read landmark coordinates
            for j in range(num_landmarks):
                i += 1
                coords = lines[i].strip().split()
                x, y = float(coords[0]), float(coords[1])
                landmarks.append((x, y))

            # Look for IMAGE= line
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('IMAGE='):
                i += 1

            if i < len(lines):
                image_name = lines[i].strip().split('=')[1]
                annotations.append((image_name, landmarks))

        i += 1

    return annotations


def create_xml_dataset(annotations: List[Tuple[str, List[Tuple[float, float]]]],
                       image_dir: Path,
                       output_path: Path):
    """Create XML file in dlib/imglab format."""

    # Create root element
    dataset = ET.Element('dataset')
    ET.SubElement(dataset, 'name').text = 'Training Dataset'
    ET.SubElement(dataset, 'comment').text = 'Generated from TPS file'

    images = ET.SubElement(dataset, 'images')

    # Add each image with annotations
    for img_name, landmarks in annotations:
        img_path = image_dir / img_name

        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}, skipping...")
            continue

        # Get image height for TPS y-axis inversion
        # TPS uses Cartesian coordinates (y=0 at bottom),
        # image coordinates have y=0 at top
        img = Image.open(img_path)
        img_w, img_h = img.size

        # Invert y: y_image = image_height - y_tps
        landmarks_img = [(x, img_h - y) for x, y in landmarks]

        # Create image element
        image = ET.SubElement(images, 'image', file=str(img_path))

        # Calculate bounding box from landmarks
        if landmarks_img:
            xs = [x for x, y in landmarks_img]
            ys = [y for x, y in landmarks_img]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Add some padding
            padding = 20
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(img_w, max_x + padding)
            max_y = min(img_h, max_y + padding)

            width = max_x - min_x
            height = max_y - min_y

            # Create box element
            box = ET.SubElement(image, 'box',
                               top=str(int(min_y)),
                               left=str(int(min_x)),
                               width=str(int(width)),
                               height=str(int(height)))

            # Add landmarks as parts
            for idx, (x, y) in enumerate(landmarks_img):
                ET.SubElement(box, 'part', name=str(idx), x=str(int(x)), y=str(int(y)))

    # Pretty print XML
    xml_str = minidom.parseString(ET.tostring(dataset)).toprettyxml(indent="   ")

    # Write to file
    with open(output_path, 'w') as f:
        f.write(xml_str)

    print(f"Created {output_path} with {len(annotations)} annotations")


def main():
    parser = argparse.ArgumentParser(description="Convert TPS to XML for PyTorch training")
    parser.add_argument('-t', '--tps', required=True, help='TPS file path')
    parser.add_argument('-i', '--images', required=True, help='Directory containing images')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    tps_path = Path(args.tps)
    image_dir = Path(args.images)

    print(f"Reading TPS file: {tps_path}")
    annotations = parse_tps_file(tps_path)
    print(f"Found {len(annotations)} annotations")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(annotations)

    split_idx = int(len(annotations) * args.train_ratio)
    train_annotations = annotations[:split_idx]
    test_annotations = annotations[split_idx:]

    print(f"\nSplit: {len(train_annotations)} train, {len(test_annotations)} test")

    # Create XML files
    create_xml_dataset(train_annotations, image_dir, Path('train.xml'))
    create_xml_dataset(test_annotations, image_dir, Path('test.xml'))

    print("\n✅ Preprocessing complete!")
    print(f"  Created: train.xml ({len(train_annotations)} images)")
    print(f"  Created: test.xml ({len(test_annotations)} images)")


if __name__ == '__main__':
    main()
