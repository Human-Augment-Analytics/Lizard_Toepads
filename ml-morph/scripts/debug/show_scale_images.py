#!/usr/bin/env python3
"""
Show actual scale images with ground truth and prediction landmarks.
"""

import dlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import xml.etree.ElementTree as ET

def read_landmarks_from_xml(xml_file, image_name):
    """Read ground truth landmarks from XML file (in bottom-left origin)."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for image_elem in root.findall('.//image'):
        file_attr = image_elem.get('file')
        if image_name in file_attr or file_attr in image_name:
            box = image_elem.find('box')
            if box is not None:
                parts = box.findall('part')
                landmarks = []
                for part in parts:
                    x = float(part.get('x'))
                    y = float(part.get('y'))
                    landmarks.append([x, y])
                if landmarks:
                    return np.array(landmarks)
    return None

def show_images(xml_file, image_dir, model_path, image_names):
    """Show actual images with landmarks."""

    # Load model
    print(f"Loading model: {model_path}")
    predictor = dlib.shape_predictor(model_path)

    for image_name in image_names:
        image_path = Path(image_dir) / image_name

        if not image_path.exists():
            print(f"Image not found: {image_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Image: {image_name}")
        print(f"{'='*60}")

        # Load image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        print(f"Image size: {w}x{h}")

        # Read ground truth from XML
        gt_landmarks = read_landmarks_from_xml(xml_file, image_name)

        # Predict landmarks
        bbox = dlib.rectangle(0, 0, w, h)
        img_dlib = dlib.load_rgb_image(str(image_path))

        try:
            shape = predictor(img_dlib, bbox)
            pred_landmarks = np.array([[p.x, p.y] for p in shape.parts()])

            # Convert from bottom-left to top-left for visualization
            pred_viz = pred_landmarks.copy()
            pred_viz[:, 1] = h - pred_landmarks[:, 1]

            print(f"\nPredicted landmarks (top-left coords):")
            for i, (x, y) in enumerate(pred_viz):
                print(f"  Point {i}: ({x:.1f}, {y:.1f})")

            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            ax.imshow(img_rgb)

            # Draw ground truth if available
            if gt_landmarks is not None:
                gt_viz = gt_landmarks.copy()
                gt_viz[:, 1] = h - gt_landmarks[:, 1]

                print(f"\nGround truth landmarks (top-left coords):")
                for i, (x, y) in enumerate(gt_viz):
                    print(f"  Point {i}: ({x:.1f}, {y:.1f})")

                # Draw GT landmarks
                for i, (x, y) in enumerate(gt_viz):
                    circle = plt.Circle((x, y), 15, color='lime', fill=True,
                                       edgecolor='black', linewidth=3, zorder=5)
                    ax.add_patch(circle)
                    ax.annotate(f'GT{i}', (x, y), fontsize=14, color='black',
                               weight='bold', ha='center', va='center', zorder=6)

                # Draw GT line
                if len(gt_viz) == 2:
                    ax.plot([gt_viz[0, 0], gt_viz[1, 0]],
                           [gt_viz[0, 1], gt_viz[1, 1]],
                           'lime', linewidth=4, alpha=0.8, zorder=4, label='Ground Truth')

                # Draw predicted landmarks
                for i, (x, y) in enumerate(pred_viz):
                    # Red X marker
                    ax.plot(x, y, 'rx', markersize=20, markeredgewidth=4, zorder=7)
                    ax.annotate(f'P{i}', (x, y-25), fontsize=12, color='red',
                               weight='bold', ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       edgecolor='red', linewidth=2), zorder=8)

                # Draw predicted line
                if len(pred_viz) == 2:
                    ax.plot([pred_viz[0, 0], pred_viz[1, 0]],
                           [pred_viz[0, 1], pred_viz[1, 1]],
                           'red', linewidth=3, alpha=0.7, zorder=3,
                           linestyle='--', label='Predicted')

                # Draw error lines
                if len(pred_viz) == len(gt_viz):
                    for i in range(len(gt_viz)):
                        ax.plot([gt_viz[i, 0], pred_viz[i, 0]],
                               [gt_viz[i, 1], pred_viz[i, 1]],
                               'yellow', linewidth=3, alpha=0.7, zorder=2)

                    # Calculate errors
                    distances = np.linalg.norm(pred_viz - gt_viz, axis=1)
                    mean_error = np.mean(distances)
                    max_error = np.max(distances)

                    print(f"\nErrors:")
                    for i, dist in enumerate(distances):
                        print(f"  Point {i}: {dist:.1f}px")
                    print(f"  Mean: {mean_error:.1f}px")
                    print(f"  Max: {max_error:.1f}px")

                    # Calculate scale lengths
                    if len(gt_viz) == 2:
                        gt_length = np.linalg.norm(gt_viz[0] - gt_viz[1])
                        pred_length = np.linalg.norm(pred_viz[0] - pred_viz[1])
                        length_error = abs(gt_length - pred_length)

                        print(f"\nScale lengths:")
                        print(f"  Ground truth: {gt_length:.1f}px")
                        print(f"  Predicted: {pred_length:.1f}px")
                        print(f"  Error: {length_error:.1f}px ({100*length_error/gt_length:.1f}%)")

                        title = (f'{image_name} ({w}x{h})\n'
                               f'Mean Error: {mean_error:.1f}px | '
                               f'GT Length: {gt_length:.1f}px | Pred: {pred_length:.1f}px | '
                               f'Length Error: {length_error:.1f}px ({100*length_error/gt_length:.1f}%)')
                    else:
                        title = f'{image_name} ({w}x{h})\nMean Error: {mean_error:.1f}px'
                else:
                    title = f'{image_name} ({w}x{h})\n(Landmark count mismatch)'
            else:
                # No ground truth
                for i, (x, y) in enumerate(pred_viz):
                    ax.plot(x, y, 'rx', markersize=20, markeredgewidth=4, zorder=7)
                    ax.annotate(f'P{i}', (x, y-25), fontsize=12, color='red',
                               weight='bold', ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       edgecolor='red', linewidth=2), zorder=8)

                if len(pred_viz) == 2:
                    ax.plot([pred_viz[0, 0], pred_viz[1, 0]],
                           [pred_viz[0, 1], pred_viz[1, 1]],
                           'red', linewidth=3, alpha=0.7, zorder=3, linestyle='--')
                    pred_length = np.linalg.norm(pred_viz[0] - pred_viz[1])
                    title = f'{image_name} ({w}x{h})\nPredicted Length: {pred_length:.1f}px (No GT)'
                else:
                    title = f'{image_name} ({w}x{h})\n(No GT available)'

            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
            ax.axis('off')

            plt.tight_layout()
            output_path = f'scale_detail_{image_name}'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved to: {output_path}")
            plt.close()

        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    xml_file = 'test.xml'
    image_dir = 'cropped_scale_padded/test'
    model_path = 'scale_predictor_fixed.dat'

    # Show the same 4 test images
    image_names = ['1017.jpg', '1036.jpg', '1044.jpg', '1053.jpg']

    show_images(xml_file, image_dir, model_path, image_names)
