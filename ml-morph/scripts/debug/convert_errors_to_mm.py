#!/usr/bin/env python3
"""
Convert hyperparameter test errors from pixels to millimeters.
Uses scale bar information from consolidated_scale.tps where the distance
between two scale points represents 10mm.
"""

import json
import math
from pathlib import Path


def parse_tps_file(tps_path):
    """
    Parse TPS file to extract scale bar information for each image.
    Returns a dictionary mapping image names to pixel-per-mm ratios.
    """
    scale_info = {}
    
    with open(tps_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    i = 0
    while i < len(lines):
        if lines[i].startswith('LM='):
            # Read the two landmark points
            x1, y1 = map(float, lines[i+1].split())
            x2, y2 = map(float, lines[i+2].split())
            
            # Calculate pixel distance between the two points
            pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Read the image name
            image_line = lines[i+3]
            if image_line.startswith('IMAGE='):
                image_name = image_line.split('=')[1]
                
                # Calculate pixels per mm (10mm scale bar)
                pixels_per_mm = pixel_distance / 10.0
                mm_per_pixel = 10.0 / pixel_distance
                
                scale_info[image_name] = {
                    'pixel_distance': pixel_distance,
                    'pixels_per_mm': pixels_per_mm,
                    'mm_per_pixel': mm_per_pixel
                }
            
            i += 4
        else:
            i += 1
    
    return scale_info


def convert_results_to_mm(results_dir, scale_info):
    """
    Convert test errors from pixels to millimeters.
    """
    results_file = Path(results_dir) / 'results.json'
    summary_file = Path(results_dir) / 'summary.json'
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Calculate average mm_per_pixel across all images
    mm_per_pixel_values = [info['mm_per_pixel'] for info in scale_info.values()]
    avg_mm_per_pixel = sum(mm_per_pixel_values) / len(mm_per_pixel_values)
    
    print(f"\nScale Statistics:")
    print(f"  Number of images with scale: {len(scale_info)}")
    print(f"  Average mm/pixel: {avg_mm_per_pixel:.6f}")
    print(f"  Min mm/pixel: {min(mm_per_pixel_values):.6f}")
    print(f"  Max mm/pixel: {max(mm_per_pixel_values):.6f}")
    
    # Convert summary results
    print(f"\n{'='*80}")
    print(f"Results for: {results_dir}")
    print(f"{'='*80}")
    
    print(f"\nBest Model: {summary['best_model']['config']}")
    print(f"  Train Error: {summary['best_model']['train_error']:.2f} pixels = {summary['best_model']['train_error'] * avg_mm_per_pixel:.4f} mm")
    print(f"  Test Error:  {summary['best_model']['test_error']:.2f} pixels = {summary['best_model']['test_error'] * avg_mm_per_pixel:.4f} mm")
    print(f"  Overfitting: {summary['best_model']['overfitting']:.2f} pixels = {summary['best_model']['overfitting'] * avg_mm_per_pixel:.4f} mm")
    
    print(f"\nTop 5 Models by Test Error:")
    print(f"{'Config':<60} {'Test (px)':<12} {'Test (mm)':<12}")
    print(f"{'-'*84}")
    for model in summary['top_5_by_test_error']:
        test_error_px = model['test_error']
        test_error_mm = test_error_px * avg_mm_per_pixel
        print(f"{model['config']:<60} {test_error_px:>10.2f}   {test_error_mm:>10.4f}")
    
    return {
        'avg_mm_per_pixel': avg_mm_per_pixel,
        'best_model_mm': {
            'train_error': summary['best_model']['train_error'] * avg_mm_per_pixel,
            'test_error': summary['best_model']['test_error'] * avg_mm_per_pixel,
            'overfitting': summary['best_model']['overfitting'] * avg_mm_per_pixel
        }
    }


def main():
    # Parse scale information
    tps_path = Path(__file__).parent / 'consolidated_scale.tps'
    print(f"Parsing scale information from: {tps_path}")
    scale_info = parse_tps_file(tps_path)
    
    # Convert finger results
    finger_dir = Path(__file__).parent / 'hyperparam_results_finger_cropped'
    finger_results = convert_results_to_mm(finger_dir, scale_info)
    
    # Convert toe results
    toe_dir = Path(__file__).parent / 'hyperparam_results_toe_cropped_fixed_yflip'
    toe_results = convert_results_to_mm(toe_dir, scale_info)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print(f"SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"\nFinger Model (Best):")
    print(f"  Test Error: {finger_results['best_model_mm']['test_error']:.4f} mm")
    print(f"\nToe Model (Best):")
    print(f"  Test Error: {toe_results['best_model_mm']['test_error']:.4f} mm")
    print(f"\nAverage mm/pixel ratio: {finger_results['avg_mm_per_pixel']:.6f}")


if __name__ == '__main__':
    main()
