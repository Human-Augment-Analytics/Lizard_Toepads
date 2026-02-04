#!/usr/bin/env python3
"""
Show a single image with predicted and ground truth landmarks overlaid.
"""

import dlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xml.etree.ElementTree as ET
import sys

def load_ground_truth_from_xml(xml_file, image_name):
    """Load ground truth landmarks from XML file."""
    if not Path(xml_file).exists():
        return None
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    images = root.findall('images/image') if root.find('images') is not None else root.findall('image')
    
    for image_elem in images:
        img_file = image_elem.get('file', '')
        xml_filename = Path(img_file).name
        if xml_filename == image_name or image_name in img_file:
            for box_elem in image_elem.findall('box'):
                landmarks_list = []
                for part_elem in box_elem.findall('part'):
                    x = float(part_elem.get('x'))
                    y = float(part_elem.get('y'))
                    landmarks_list.append([x, y])
                
                if landmarks_list:
                    return np.array(landmarks_list)
    return None


def visualize_single_image(image_path, model_path, xml_file, output_path=None):
    """Visualize predictions vs ground truth for a single image."""
    # Load predictor
    predictor = dlib.shape_predictor(model_path)
    
    # Load image
    img = dlib.load_rgb_image(str(image_path))
    img_h, img_w = img.shape[:2]
    
    # Get ground truth
    image_name = Path(image_path).name
    gt_landmarks = load_ground_truth_from_xml(xml_file, image_name)
    
    # Use full image as bounding box for cropped images
    bbox = dlib.rectangle(0, 0, img_w, img_h)
    
    # Predict landmarks
    try:
        pred_shape = predictor(img, bbox)
        pred_landmarks = np.array([[p.x, p.y] for p in pred_shape.parts()])
        
        # Convert predicted landmarks from bottom-left to top-left for visualization
        pred_landmarks_viz = pred_landmarks.copy()
        pred_landmarks_viz[:, 1] = img_h - pred_landmarks[:, 1]
        
    except Exception as e:
        print(f"Error predicting {image_name}: {e}")
        return
    
    # Convert ground truth if available
    gt_landmarks_viz = None
    if gt_landmarks is not None:
        gt_landmarks_viz = gt_landmarks.copy()
        gt_landmarks_viz[:, 1] = img_h - gt_landmarks[:, 1]
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)
    
    # Draw ground truth landmarks first (green, behind)
    if gt_landmarks_viz is not None:
        ax.scatter(gt_landmarks_viz[:, 0], gt_landmarks_viz[:, 1], 
                  c='lime', s=200, marker='x', 
                  label='Ground Truth', zorder=4, linewidths=4)
        
        # Draw line connecting ground truth landmarks
        if len(gt_landmarks_viz) >= 2:
            ax.plot(gt_landmarks_viz[:, 0], gt_landmarks_viz[:, 1], 
                   'g-', linewidth=3, alpha=0.7, zorder=2, label='GT Line')
    
    # Draw predicted landmarks (red, on top)
    if pred_landmarks_viz is not None:
        ax.scatter(pred_landmarks_viz[:, 0], pred_landmarks_viz[:, 1], 
                  c='red', s=200, marker='o', 
                  label='Predicted', zorder=5, edgecolors='darkred', linewidths=2)
        
        # Draw line connecting predicted landmarks
        if len(pred_landmarks_viz) >= 2:
            ax.plot(pred_landmarks_viz[:, 0], pred_landmarks_viz[:, 1], 
                   'r-', linewidth=3, alpha=0.7, zorder=3, label='Pred Line')
    
    # Draw error lines connecting corresponding landmarks
    if gt_landmarks_viz is not None and pred_landmarks_viz is not None:
        if len(pred_landmarks_viz) == len(gt_landmarks_viz):
            for i in range(len(gt_landmarks_viz)):
                ax.plot([gt_landmarks_viz[i, 0], pred_landmarks_viz[i, 0]], 
                       [gt_landmarks_viz[i, 1], pred_landmarks_viz[i, 1]], 
                       'yellow', linewidth=2, alpha=0.5, zorder=1)
            
            # Calculate and display error metrics
            distances = np.linalg.norm(pred_landmarks_viz - gt_landmarks_viz, axis=1)
            mean_error = np.mean(distances)
            max_error = np.max(distances)
            min_error = np.min(distances)
            std_error = np.std(distances)
            
            # Add text box with error metrics
            error_text = f'Mean Error: {mean_error:.2f} px\n'
            error_text += f'Max Error: {max_error:.2f} px\n'
            error_text += f'Min Error: {min_error:.2f} px\n'
            error_text += f'Std Error: {std_error:.2f} px'
            
            ax.text(0.02, 0.98, error_text, 
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   zorder=6)
            
            title = f'{image_name} - Scale Landmark Prediction'
        else:
            title = f'{image_name} - Landmark count mismatch (GT: {len(gt_landmarks_viz)}, Pred: {len(pred_landmarks_viz)})'
    else:
        title = f'{image_name} - No ground truth available'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Show single image with predictions vs ground truth')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='scale_predictor_cropped.dat', help='Path to model')
    parser.add_argument('--xml', type=str, default='test.xml', help='Path to XML file')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    
    args = parser.parse_args()
    
    visualize_single_image(args.image, args.model, args.xml, args.output)
