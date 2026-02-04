#!/usr/bin/env python3
"""
Investigate why test error is high (21.64 vs expected 8.77).
Check coordinate systems, visualize errors, and compare with previous results.
"""

import dlib
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_xml(xml_path):
    """Parse dlib XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    images = root.findall('images/image') if root.find('images') is not None else root.findall('image')
    
    data = []
    for image_elem in images:
        image_file = image_elem.get('file')
        boxes = image_elem.findall('box')
        
        for box in boxes:
            top = int(box.get('top'))
            left = int(box.get('left'))
            width = int(box.get('width'))
            height = int(box.get('height'))
            
            parts = []
            for part in box.findall('part'):
                x = int(part.get('x'))
                y = int(part.get('y'))
                parts.append((x, y))
            
            data.append({
                'image': image_file,
                'bbox': (left, top, width, height),
                'landmarks': np.array(parts)
            })
    
    return data

def compute_errors(test_xml, model_path, image_dir):
    """Compute per-image errors."""
    predictor = dlib.shape_predictor(model_path)
    test_data = parse_xml(test_xml)
    
    errors = []
    
    for item in test_data:
        image_path = None
        # Try to find image
        possible_paths = [
            Path(item['image']),
            Path(image_dir) / Path(item['image']).name,
            Path(image_dir) / item['image'],
        ]
        
        for path in possible_paths:
            if path.exists():
                image_path = path
                break
        
        if image_path is None:
            print(f"Warning: Could not find image {item['image']}")
            continue
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not load image {image_path}")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Get ground truth landmarks
        gt_landmarks = item['landmarks']
        
        # Create dlib rectangle from bbox
        left, top, width, height = item['bbox']
        bbox = dlib.rectangle(left, top, left + width, top + height)
        
        # Predict landmarks
        pred_shape = predictor(img, bbox)
        pred_landmarks = np.array([[p.x, p.y] for p in pred_shape.parts()])
        
        # Compute error
        if len(pred_landmarks) != len(gt_landmarks):
            print(f"Warning: Mismatch in landmark count for {item['image']}")
            continue
        
        # Calculate per-landmark errors
        per_landmark_errors = np.sqrt(np.sum((pred_landmarks - gt_landmarks) ** 2, axis=1))
        mean_error = np.mean(per_landmark_errors)
        max_error = np.max(per_landmark_errors)
        
        errors.append({
            'image': item['image'],
            'mean_error': mean_error,
            'max_error': max_error,
            'per_landmark_errors': per_landmark_errors,
            'gt_landmarks': gt_landmarks,
            'pred_landmarks': pred_landmarks,
            'bbox': item['bbox'],
            'img_size': (img_width, img_height)
        })
    
    return errors

def analyze_errors(errors):
    """Analyze error distribution."""
    mean_errors = [e['mean_error'] for e in errors]
    max_errors = [e['max_error'] for e in errors]
    
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    print(f"Total images: {len(errors)}")
    print(f"\nMean Error Statistics:")
    print(f"  Mean: {np.mean(mean_errors):.2f} pixels")
    print(f"  Median: {np.median(mean_errors):.2f} pixels")
    print(f"  Std: {np.std(mean_errors):.2f} pixels")
    print(f"  Min: {np.min(mean_errors):.2f} pixels")
    print(f"  Max: {np.max(mean_errors):.2f} pixels")
    
    print(f"\nMax Error Statistics:")
    print(f"  Mean: {np.mean(max_errors):.2f} pixels")
    print(f"  Median: {np.median(max_errors):.2f} pixels")
    print(f"  Std: {np.std(max_errors):.2f} pixels")
    print(f"  Min: {np.min(max_errors):.2f} pixels")
    print(f"  Max: {np.max(max_errors):.2f} pixels")
    
    # Find outliers
    q75 = np.percentile(mean_errors, 75)
    q25 = np.percentile(mean_errors, 25)
    iqr = q75 - q25
    outlier_threshold = q75 + 1.5 * iqr
    
    outliers = [e for e in errors if e['mean_error'] > outlier_threshold]
    print(f"\nOutliers (>{outlier_threshold:.2f} pixels): {len(outliers)}")
    
    # Show worst errors
    sorted_errors = sorted(errors, key=lambda x: x['mean_error'], reverse=True)
    print(f"\nTop 5 Worst Errors:")
    for i, err in enumerate(sorted_errors[:5], 1):
        print(f"  {i}. {err['image']}: {err['mean_error']:.2f} pixels (max: {err['max_error']:.2f})")
    
    return sorted_errors

def visualize_worst_errors(errors, output_dir='error_analysis', num_images=5):
    """Visualize worst errors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    sorted_errors = sorted(errors, key=lambda x: x['mean_error'], reverse=True)
    
    for i, err in enumerate(sorted_errors[:num_images], 1):
        # Try to load image
        image_path = None
        possible_paths = [
            Path(err['image']),
            Path('test') / Path(err['image']).name,
            Path('train') / Path(err['image']).name,
        ]
        
        for path in possible_paths:
            if path.exists():
                image_path = path
                break
        
        if image_path is None:
            print(f"Warning: Could not find image for {err['image']}")
            continue
        
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(img_rgb)
        
        # Draw bbox
        left, top, width, height = err['bbox']
        rect = plt.Rectangle((left, top), width, height, 
                           fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
        
        # Draw GT landmarks (green)
        gt = err['gt_landmarks']
        ax.scatter(gt[:, 0], gt[:, 1], c='green', s=50, marker='o', 
                  label='Ground Truth', zorder=5, edgecolors='black', linewidths=1)
        
        # Draw predicted landmarks (red)
        pred = err['pred_landmarks']
        ax.scatter(pred[:, 0], pred[:, 1], c='red', s=50, marker='x', 
                  label='Predicted', zorder=5, linewidths=2)
        
        # Draw error lines
        for j in range(len(gt)):
            ax.plot([gt[j, 0], pred[j, 0]], [gt[j, 1], pred[j, 1]], 
                   'yellow', linewidth=1, alpha=0.5, zorder=3)
            ax.annotate(f'{err["per_landmark_errors"][j]:.1f}', 
                       (pred[j, 0], pred[j, 1]), 
                       fontsize=8, color='yellow', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        ax.set_title(f'{Path(err["image"]).name}\n'
                    f'Mean Error: {err["mean_error"]:.2f}px, Max: {err["max_error"]:.2f}px',
                    fontsize=12)
        ax.legend()
        ax.axis('off')
        
        output_path = output_dir / f'worst_error_{i:02d}_{Path(err["image"]).stem}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Investigate test error')
    parser.add_argument('--test-xml', default='test.xml', help='Test XML file')
    parser.add_argument('--model', default='toe_predictor_full.dat', help='Model path')
    parser.add_argument('--image-dir', default='test', help='Image directory')
    parser.add_argument('--output-dir', default='error_analysis', help='Output directory')
    parser.add_argument('--num-visualize', type=int, default=5, help='Number of worst errors to visualize')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("INVESTIGATING TEST ERROR")
    print("=" * 60)
    print(f"Test XML: {args.test_xml}")
    print(f"Model: {args.model}")
    print(f"Image directory: {args.image_dir}")
    print("=" * 60)
    
    # Compute errors
    print("\nComputing errors...")
    errors = compute_errors(args.test_xml, args.model, args.image_dir)
    
    if not errors:
        print("No errors computed! Check paths.")
        return
    
    # Analyze
    sorted_errors = analyze_errors(errors)
    
    # Visualize
    print(f"\nVisualizing worst {args.num_visualize} errors...")
    visualize_worst_errors(sorted_errors, args.output_dir, args.num_visualize)
    
    print(f"\n✓ Analysis complete! Check {args.output_dir}/ for visualizations.")

if __name__ == '__main__':
    main()

