#!/usr/bin/env python3
"""
Train shape predictor on cropped finger images using best parameters from hyperparameter search.

Best parameters from hyperparam_results_finger:
- tree_depth: 3
- cascade_depth: 10
- nu: 0.05
- num_trees: 500
- oversampling: 10
- feature_pool_size: 500
- test_splits: 20
- threads: 4
"""

import argparse
import os
import sys
import dlib

# Best parameters from hyperparam_results_finger/summary.json
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
    print("Training Shape Predictor with Best Parameters")
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
    options.num_threads = params['threads']
    options.be_verbose = True
    
    # Train the model
    print(f"\nTraining on: {train_xml}")
    print(f"Output model: {output_file}")
    print("\nStarting training...\n")
    
    try:
        dlib.train_shape_predictor(train_xml, output_file, options)
        
        # Evaluate on training set
        train_error = dlib.test_shape_predictor(train_xml, output_file)
        print(f"\n{'=' * 60}")
        print(f"Training complete!")
        print(f"Training error (average pixel deviation): {train_error:.6f}")
        
        # Evaluate on test set if provided
        if test_xml and os.path.exists(test_xml):
            test_error = dlib.test_shape_predictor(test_xml, output_file)
            print(f"Test error (average pixel deviation): {test_error:.6f}")
            print(f"Overfitting (test - train): {test_error - train_error:.6f}")
        
        print(f"{'=' * 60}")
        print(f"Model saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Train shape predictor on cropped finger images with best parameters'
    )
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        default='train.xml',
        help='Training data XML file (default: train.xml)'
    )
    parser.add_argument(
        '-t', '--test',
        type=str,
        default='test.xml',
        help='Test data XML file (default: test.xml)'
    )
    parser.add_argument(
        '-o', '--out',
        type=str,
        default='finger_predictor_cropped.dat',
        help='Output model filename (default: finger_predictor_cropped.dat)'
    )
    parser.add_argument(
        '--no-test',
        action='store_true',
        help='Skip testing (do not evaluate on test set)'
    )
    
    # Optional parameter overrides
    parser.add_argument(
        '--tree-depth',
        type=int,
        default=None,
        help=f'Tree depth (default: {BEST_PARAMS["tree_depth"]})'
    )
    parser.add_argument(
        '--cascade-depth',
        type=int,
        default=None,
        help=f'Cascade depth (default: {BEST_PARAMS["cascade_depth"]})'
    )
    parser.add_argument(
        '--nu',
        type=float,
        default=None,
        help=f'Regularization parameter (default: {BEST_PARAMS["nu"]})'
    )
    parser.add_argument(
        '--num-trees',
        type=int,
        default=None,
        help=f'Number of trees (default: {BEST_PARAMS["num_trees"]})'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=None,
        help=f'Number of threads (default: {BEST_PARAMS["threads"]})'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataset):
        print(f"Error: Training XML file not found: {args.dataset}")
        sys.exit(1)
    
    test_xml = None if args.no_test else args.test
    if test_xml and not os.path.exists(test_xml):
        print(f"Warning: Test XML file not found: {test_xml}")
        print("Continuing without test evaluation...")
        test_xml = None
    
    # Build parameters dict (use overrides if provided)
    params = BEST_PARAMS.copy()
    if args.tree_depth is not None:
        params['tree_depth'] = args.tree_depth
    if args.cascade_depth is not None:
        params['cascade_depth'] = args.cascade_depth
    if args.nu is not None:
        params['nu'] = args.nu
    if args.num_trees is not None:
        params['num_trees'] = args.num_trees
    if args.threads is not None:
        params['threads'] = args.threads
    
    # Train
    train_shape_predictor(
        train_xml=args.dataset,
        test_xml=test_xml,
        output_file=args.out,
        params=params
    )


if __name__ == '__main__':
    main()


