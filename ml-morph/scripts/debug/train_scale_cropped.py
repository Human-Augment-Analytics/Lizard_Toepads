#!/usr/bin/env python3
"""
Train shape predictor on cropped scale images using YOLO-cropped images.

This script trains a dlib shape predictor on scale landmarks using images
that were cropped using YOLO bounding box detection.
"""

import dlib
import argparse
import os
import json
from pathlib import Path

def train_shape_predictor(train_xml, test_xml, output_file, params=None):
    """
    Train shape predictor with given parameters.
    If params is None, uses best parameters from hyperparameter search.
    """
    if params is None:
        # Best parameters from hyperparameter search
        params = {
            'tree_depth': 3,
            'cascade_depth': 15,
            'nu': 0.1,
            'num_trees': 500,
            'oversampling': 10,
            'feature_pool_size': 500,
            'test_splits': 20,
            'threads': 4
        }
    
    print("Training Shape Predictor for Scale Landmarks")
    print("=" * 60)
    for key, value in params.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Set up training options
    options = dlib.shape_predictor_training_options()
    options.tree_depth = params['tree_depth']
    options.cascade_depth = params['cascade_depth']
    options.nu = params['nu']
    options.num_trees_per_cascade_level = params['num_trees']
    options.oversampling_amount = params['oversampling']
    options.feature_pool_size = params['feature_pool_size']
    options.num_test_splits = params['test_splits']
    options.be_verbose = True
    
    # Train the shape predictor
    print(f"\nTraining on: {train_xml}")
    print(f"Output: {output_file}")
    print("\nStarting training...")
    
    try:
        dlib.train_shape_predictor(train_xml, output_file, options)
        
        # Evaluate on training set
        train_error = dlib.test_shape_predictor(train_xml, output_file)
        print(f"\nTraining error (average pixel deviation): {train_error:.6f}")
        
        # Evaluate on test set if provided
        if test_xml and os.path.exists(test_xml):
            test_error = dlib.test_shape_predictor(test_xml, output_file)
            print(f"Test error (average pixel deviation): {test_error:.6f}")
            print(f"Overfitting (test - train): {test_error - train_error:.6f}")
        
        print(f"\n✓ Training complete! Model saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description='Train shape predictor on cropped scale images with YOLO-cropped images'
    )
    parser.add_argument(
        '--train-xml',
        type=str,
        default='train.xml',
        help='Path to training XML file (default: train.xml)'
    )
    parser.add_argument(
        '--test-xml',
        type=str,
        default='test.xml',
        help='Path to test XML file (default: test.xml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='scale_predictor_cropped.dat',
        help='Output model filename (default: scale_predictor_cropped.dat)'
    )
    parser.add_argument(
        '--params-file',
        type=str,
        default=None,
        help='Path to JSON file with parameters (optional)'
    )
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip testing (do not evaluate on test set)'
    )
    
    args = parser.parse_args()
    
    # Load parameters if provided
    params = None
    if args.params_file and os.path.exists(args.params_file):
        with open(args.params_file, 'r') as f:
            data = json.load(f)
            if 'best_model' in data:
                params = data['best_model']['params']
            elif 'params' in data:
                params = data['params']
            else:
                params = data
        print(f"Loaded parameters from: {args.params_file}")
    
    # Train
    train_shape_predictor(
        args.train_xml,
        None if args.skip_test else args.test_xml,
        args.output,
        params
    )

if __name__ == '__main__':
    main()
