#!/usr/bin/env python3
"""
Train shape predictor on full (non-cropped) toe images using best parameters from hyperparameter search.

Best parameters from hyperparam_results_toe:
- tree_depth: 3
- cascade_depth: 10
- nu: 0.05
- num_trees: 500
- oversampling: 10
- feature_pool_size: 500
- test_splits: 20
- threads: 4

Test error: ~8.77 pixels (from previous non-cropped training)
"""

import dlib
import argparse
import os
import json
from pathlib import Path

# Best parameters from hyperparam_results_toe (lowest test error)
BEST_PARAMS = {
    'tree_depth': 3,
    'cascade_depth': 10,
    'nu': 0.05,
    'num_trees': 500,
    'oversampling': 10,
    'feature_pool_size': 500,
    'test_splits': 20,
    'threads': 4
}

def train_shape_predictor(train_xml, test_xml, output_file, params=None):
    """
    Train shape predictor with given parameters.
    
    Args:
        train_xml: Path to training XML file
        test_xml: Path to test XML file
        output_file: Output path for trained model (.dat file)
        params: Dictionary of training parameters (uses BEST_PARAMS if None)
    """
    if params is None:
        params = BEST_PARAMS
    
    print("=" * 60)
    print("Training Shape Predictor on Full Toe Images")
    print("=" * 60)
    print(f"Tree depth: {params['tree_depth']}")
    print(f"Cascade depth: {params['cascade_depth']}")
    print(f"Nu (regularization): {params['nu']}")
    print(f"Number of trees: {params['num_trees']}")
    print(f"Oversampling: {params['oversampling']}")
    print(f"Feature pool size: {params['feature_pool_size']}")
    print(f"Test splits: {params['test_splits']}")
    print(f"Threads: {params['threads']}")
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
    print("(This may take a while...)")
    
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
        description='Train shape predictor on full (non-cropped) toe images with best parameters'
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
        default='toe_predictor_full.dat',
        help='Output model filename (default: toe_predictor_full.dat)'
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
    
    # Validate XML files exist
    if not os.path.exists(args.train_xml):
        print(f"Error: Training XML file not found: {args.train_xml}")
        print("Run preprocess_toe_full.py first to generate train.xml and test.xml")
        return 1
    
    # Train
    train_shape_predictor(
        args.train_xml,
        None if args.skip_test else args.test_xml,
        args.output,
        params
    )
    
    return 0

if __name__ == '__main__':
    exit(main())

