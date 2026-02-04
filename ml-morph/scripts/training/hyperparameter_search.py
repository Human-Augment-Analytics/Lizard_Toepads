#!/usr/bin/env python3
"""
Hyperparameter search for ml-morph shape predictor training.
Searches for optimal parameters to minimize test error while avoiding overfitting.
"""
import os
import json
import argparse
from itertools import product
from datetime import datetime
import dlib


def make_config_name(params):
    """
    Build a unique filename component for a given parameter combination.
    Ensures variations in oversampling and other options do not collide.
    """
    parts = [
        f"depth{params['tree_depth']}",
        f"cascade{params['cascade_depth']}",
        f"nu{params['nu']}",
        f"trees{params['num_trees']}",
        f"over{params['oversampling']}",
        f"fp{params['feature_pool_size']}",
        f"splits{params['test_splits']}",
    ]
    return "_".join(parts)


def train_and_evaluate(train_xml, test_xml, params, output_dir='hyperparam_results'):
    """
    Train a shape predictor with given parameters and evaluate.

    Returns:
        dict: Results including train error, test error, and parameters
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalize parameters so config naming includes defaults
    normalized_params = params.copy()
    normalized_params.setdefault('oversampling', 10)
    normalized_params.setdefault('feature_pool_size', 500)
    normalized_params.setdefault('test_splits', 20)
    normalized_params.setdefault('threads', 4)

    # Create unique name for this configuration
    config_name = make_config_name(normalized_params)
    output_path = os.path.join(output_dir, f"{config_name}.dat")

    # Skip if already trained
    if os.path.exists(output_path):
        print(f"Skipping {config_name} - already exists")
        train_error = dlib.test_shape_predictor(train_xml, output_path)
        test_error = dlib.test_shape_predictor(test_xml, output_path)
        return {
            'config': config_name,
            'params': normalized_params,
            'train_error': train_error,
            'test_error': test_error,
            'overfitting': test_error - train_error,
            'model_path': output_path
        }

    print(f"\nTraining: {config_name}")
    print(f"Parameters: {params}")

    # Set up training options
    options = dlib.shape_predictor_training_options()
    options.tree_depth = normalized_params['tree_depth']
    options.cascade_depth = normalized_params['cascade_depth']
    options.nu = normalized_params['nu']
    options.num_trees_per_cascade_level = normalized_params['num_trees']
    options.oversampling_amount = normalized_params['oversampling']
    options.feature_pool_size = normalized_params['feature_pool_size']
    options.num_test_splits = normalized_params['test_splits']
    options.num_threads = normalized_params['threads']
    options.be_verbose = True

    # Train the model
    try:
        dlib.train_shape_predictor(train_xml, output_path, options)

        # Evaluate on training and test sets
        train_error = dlib.test_shape_predictor(train_xml, output_path)
        test_error = dlib.test_shape_predictor(test_xml, output_path)

        result = {
            'config': config_name,
            'params': normalized_params,
            'train_error': train_error,
            'test_error': test_error,
            'overfitting': test_error - train_error,
            'model_path': output_path
        }

        print(f"Train Error: {train_error:.4f}")
        print(f"Test Error: {test_error:.4f}")
        print(f"Overfitting: {test_error - train_error:.4f}")

        return result

    except Exception as e:
        print(f"Error training {config_name}: {e}")
        return None


def build_param_grid(args):
    """
    Construct the parameter grid based on CLI flags.
    """
    if args.refine:
        return {
            'tree_depth': [2, 3],
            'cascade_depth': [8, 10, 12],
            'nu': [0.05, 0.075, 0.1, 0.125, 0.15],
            'num_trees': [500, 600, 700, 800],
            'oversampling': [8, 12, 16],
            'feature_pool_size': [500, 700],
            'test_splits': [20, 30],
            'threads': [args.threads],
        }
    if args.quick:
        return {
            'tree_depth': [3, 4, 5],
            'cascade_depth': [10, 15],
            'nu': [0.05, 0.1, 0.2],
            'num_trees': [300, 500],
            'oversampling': [10],
            'feature_pool_size': [500],
            'test_splits': [20],
            'threads': [args.threads],
        }
    # Full search
    return {
        'tree_depth': [2, 3, 4, 5, 6],
        'cascade_depth': [8, 10, 12, 15],
        'nu': [0.01, 0.05, 0.1, 0.15, 0.2],
        'num_trees': [200, 300, 500, 700],
        'oversampling': [5, 10, 20],
        'feature_pool_size': [500],
        'test_splits': [20],
        'threads': [args.threads],
    }


def grid_search(train_xml, test_xml, param_grid, output_dir='hyperparam_results'):
    """
    Perform grid search over hyperparameter space.

    Args:
        train_xml: Path to training XML
        test_xml: Path to test XML
        param_grid: Dictionary of parameter lists to search

    Returns:
        list: Results for all configurations
    """
    results = []

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    total_configs = 1
    for v in values:
        total_configs *= len(v)

    print(f"Testing {total_configs} configurations...")

    for i, combination in enumerate(product(*values), 1):
        params = dict(zip(keys, combination))
        print(f"\n{'='*60}")
        print(f"Configuration {i}/{total_configs}")
        print(f"{'='*60}")

        result = train_and_evaluate(train_xml, test_xml, params, output_dir)
        if result:
            results.append(result)

            # Save intermediate results
            results_file = os.path.join(output_dir, 'results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

    return results


def analyze_results(results, output_dir='hyperparam_results'):
    """
    Analyze and summarize hyperparameter search results.
    """
    if not results:
        print("No results to analyze")
        return

    # Sort by test error
    sorted_by_test = sorted(results, key=lambda x: x['test_error'])

    # Sort by overfitting (smallest gap)
    sorted_by_overfit = sorted(results, key=lambda x: x['overfitting'])

    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*80)

    print("\n📊 TOP 5 MODELS BY TEST ERROR:")
    print("-" * 80)
    for i, r in enumerate(sorted_by_test[:5], 1):
        print(f"\n{i}. {r['config']}")
        print(f"   Train Error: {r['train_error']:.4f} | Test Error: {r['test_error']:.4f} | Overfitting: {r['overfitting']:.4f}")
        print(f"   Params: {r['params']}")

    print("\n\n🎯 TOP 5 MODELS BY SMALLEST OVERFITTING:")
    print("-" * 80)
    for i, r in enumerate(sorted_by_overfit[:5], 1):
        print(f"\n{i}. {r['config']}")
        print(f"   Train Error: {r['train_error']:.4f} | Test Error: {r['test_error']:.4f} | Overfitting: {r['overfitting']:.4f}")
        print(f"   Params: {r['params']}")

    print("\n\n⭐ BEST OVERALL MODEL (Lowest Test Error):")
    print("-" * 80)
    best = sorted_by_test[0]
    print(f"Config: {best['config']}")
    print(f"Train Error: {best['train_error']:.4f}")
    print(f"Test Error: {best['test_error']:.4f}")
    print(f"Overfitting: {best['overfitting']:.4f}")
    print(f"Parameters: {json.dumps(best['params'], indent=2)}")
    print(f"Model Path: {best['model_path']}")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_configs': len(results),
        'best_model': best,
        'top_5_by_test_error': sorted_by_test[:5],
        'top_5_by_overfitting': sorted_by_overfit[:5]
    }

    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Results saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search for shape predictor')
    parser.add_argument('--train', default='train.xml', help='Training XML file')
    parser.add_argument('--test', default='test.xml', help='Test XML file')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: hyperparam_results_<category>)')
    parser.add_argument('--category', default='finger', choices=['finger', 'toe', 'id', 'scale'], 
                        help='Category name for organizing results (default: finger)')
    parser.add_argument('--quick', action='store_true', help='Quick search with fewer parameters')
    parser.add_argument('--refine', action='store_true', help='Focused search around best-performing configurations')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')

    args = parser.parse_args()
    
    # Set default output directory based on category if not specified
    if args.output_dir is None:
        args.output_dir = f'hyperparam_results_{args.category}'

    print(f"Starting hyperparameter search for {args.category}...")
    print(f"Training data: {args.train}")
    print(f"Test data: {args.test}")
    print(f"Output directory: {args.output_dir}")
    if args.refine and args.quick:
        raise ValueError("Use only one of --quick or --refine.")
    mode = 'Refine' if args.refine else ('Quick' if args.quick else 'Full')
    print(f"Mode: {mode}")

    param_grid = build_param_grid(args)

    # Run grid search
    results = grid_search(args.train, args.test, param_grid, args.output_dir)

    # Analyze results
    analyze_results(results, args.output_dir)


if __name__ == '__main__':
    main()
