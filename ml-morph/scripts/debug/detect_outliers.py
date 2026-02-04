#!/usr/bin/env python3
"""
Detect outliers in the dataset using the best model from hyperparameter search results.
Computes per-sample prediction errors and identifies outliers using multiple methods.
"""
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import dlib

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Note: scipy not available. Z-score calculation will use numpy.")

try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Note: scikit-learn not available. Isolation Forest method will be skipped.")


def load_best_model(results_json_path: str) -> Dict[str, Any]:
    """
    Load the best model from results.json (lowest test_error).
    
    Args:
        results_json_path: Path to results.json file
        
    Returns:
        Dictionary with best model information
    """
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    # Find model with lowest test_error
    best_model = min(results, key=lambda x: x['test_error'])
    
    print(f"Best model: {best_model['config']}")
    print(f"  Test Error: {best_model['test_error']:.4f}")
    print(f"  Train Error: {best_model['train_error']:.4f}")
    print(f"  Model Path: {best_model['model_path']}")
    
    return best_model


def parse_xml(xml_path: Path) -> List[Dict]:
    """
    Parse dlib XML file and extract image annotations.
    
    Args:
        xml_path: Path to XML file
        
    Returns:
        List of dictionaries with image file, bbox, and keypoints
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    samples = []
    for image_elem in root.findall('.//image'):
        image_file = image_elem.get('file')
        
        for box_elem in image_elem.findall('box'):
            top = int(box_elem.get('top'))
            left = int(box_elem.get('left'))
            width = int(box_elem.get('width'))
            height = int(box_elem.get('height'))
            
            # Extract keypoints
            keypoints = []
            for part in box_elem.findall('part'):
                x = float(part.get('x'))
                y = float(part.get('y'))
                keypoints.append((x, y))
            
            samples.append({
                'image_file': image_file,
                'bbox': dlib.rectangle(left, top, left + width, top + height),
                'keypoints': np.array(keypoints, dtype=np.float64)
            })
    
    return samples


def compute_per_sample_errors(
    xml_path: Path,
    model_path: str,
    root_dir: Path
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Compute prediction errors for each sample.
    
    Args:
        xml_path: Path to XML file with ground truth
        model_path: Path to trained shape predictor
        root_dir: Root directory for images
        
    Returns:
        Tuple of (image_files, mean_errors, max_errors)
        - image_files: List of image file paths
        - mean_errors: Mean error per sample (across all keypoints)
        - max_errors: Max error per sample (worst keypoint error)
    """
    # Load predictor
    predictor = dlib.shape_predictor(model_path)
    
    # Parse XML
    samples = parse_xml(xml_path)
    
    image_files = []
    mean_errors = []
    max_errors = []
    per_sample_keypoint_errors = []
    
    print(f"\nComputing errors for {len(samples)} samples...")
    
    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples...")
        
        image_path = root_dir / sample['image_file']
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Load image
        img = dlib.load_rgb_image(str(image_path))
        
        # Get bounding box and ground truth keypoints
        bbox = sample['bbox']
        gt_keypoints = sample['keypoints']
        
        # Make prediction
        try:
            pred_shape = predictor(img, bbox)
            pred_keypoints = np.array([[p.x, p.y] for p in pred_shape.parts()], dtype=np.float64)
        except Exception as e:
            print(f"Warning: Prediction failed for {image_path}: {e}")
            continue
        
        # Handle mismatch in number of keypoints
        if len(pred_keypoints) != len(gt_keypoints):
            min_len = min(len(pred_keypoints), len(gt_keypoints))
            pred_keypoints = pred_keypoints[:min_len]
            gt_keypoints = gt_keypoints[:min_len]
        
        # Compute Euclidean distances for each keypoint
        distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)
        
        # Store results
        image_files.append(str(sample['image_file']))
        mean_errors.append(np.mean(distances))
        max_errors.append(np.max(distances))
        per_sample_keypoint_errors.append(distances)
    
    return (
        image_files,
        np.array(mean_errors),
        np.array(max_errors),
        np.array(per_sample_keypoint_errors)
    )


def detect_outliers_iqr(errors: np.ndarray, k: float = 1.5) -> np.ndarray:
    """
    Detect outliers using IQR (Interquartile Range) method.
    
    Args:
        errors: Array of error values
        k: Multiplier for IQR (default 1.5)
        
    Returns:
        Boolean array where True indicates outlier
    """
    Q1 = np.percentile(errors, 25)
    Q3 = np.percentile(errors, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    return (errors < lower_bound) | (errors > upper_bound)


def detect_outliers_zscore(errors: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers using Z-score method.
    
    Args:
        errors: Array of error values
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        Boolean array where True indicates outlier
    """
    if HAS_SCIPY:
        z_scores = np.abs(stats.zscore(errors))
    else:
        # Manual z-score calculation
        mean = np.mean(errors)
        std = np.std(errors)
        if std == 0:
            return np.zeros(len(errors), dtype=bool)
        z_scores = np.abs((errors - mean) / std)
    return z_scores > threshold


def detect_outliers_percentile(errors: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """
    Detect outliers using percentile method.
    
    Args:
        errors: Array of error values
        lower: Lower percentile threshold (default 1.0)
        upper: Upper percentile threshold (default 99.0)
        
    Returns:
        Boolean array where True indicates outlier
    """
    lower_bound = np.percentile(errors, lower)
    upper_bound = np.percentile(errors, upper)
    
    return (errors < lower_bound) | (errors > upper_bound)


def detect_outliers_isolation_forest(errors: np.ndarray, contamination: float = 0.1) -> np.ndarray:
    """
    Detect outliers using Isolation Forest.
    
    Args:
        errors: Array of error values (reshaped to 2D)
        contamination: Expected proportion of outliers (default 0.1)
        
    Returns:
        Boolean array where True indicates outlier (-1 in Isolation Forest)
    """
    if not HAS_SKLEARN:
        return np.zeros(len(errors), dtype=bool)
    
    # Reshape for sklearn
    X = errors.reshape(-1, 1)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(X)
    
    # Return True for outliers (predictions == -1)
    return predictions == -1


def main():
    parser = argparse.ArgumentParser(
        description='Detect outliers in dataset using best model from hyperparameter search'
    )
    parser.add_argument(
        '--results',
        type=str,
        default='ml-morph/hyperparam_results/results.json',
        help='Path to results.json file'
    )
    parser.add_argument(
        '--xml',
        type=str,
        default='ml-morph/test.xml',
        help='Path to test XML file (or train.xml)'
    )
    parser.add_argument(
        '--root',
        type=str,
        default='ml-morph',
        help='Root directory for images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ml-morph/outliers_results.json',
        help='Output file for outlier results'
    )
    parser.add_argument(
        '--use-train',
        action='store_true',
        help='Use training set instead of test set'
    )
    
    args = parser.parse_args()
    
    # Determine which XML to use
    if args.use_train:
        xml_path = Path('ml-morph/train.xml')
    else:
        xml_path = Path(args.xml)
    
    root_dir = Path(args.root)
    results_json = Path(args.results)
    output_path = Path(args.output)
    
    # Load best model
    print("=" * 80)
    print("OUTLIER DETECTION")
    print("=" * 80)
    best_model = load_best_model(str(results_json))
    
    # Fix model path (handle Windows backslashes)
    model_path = Path(best_model['model_path']).resolve()
    if not model_path.exists():
        # Try relative path
        model_path = Path(best_model['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {best_model['model_path']}")
    
    print(f"\nUsing model: {model_path}")
    print(f"Using XML: {xml_path}")
    
    # Compute per-sample errors
    image_files, mean_errors, max_errors, keypoint_errors = compute_per_sample_errors(
        xml_path, str(model_path), root_dir
    )
    
    if len(mean_errors) == 0:
        print("Error: No samples processed successfully!")
        return
    
    print(f"\nComputed errors for {len(mean_errors)} samples")
    print(f"Mean error: {np.mean(mean_errors):.4f} ± {np.std(mean_errors):.4f}")
    print(f"Median error: {np.median(mean_errors):.4f}")
    print(f"Min error: {np.min(mean_errors):.4f}")
    print(f"Max error: {np.max(mean_errors):.4f}")
    
    # Detect outliers using multiple methods
    print("\n" + "=" * 80)
    print("OUTLIER DETECTION RESULTS")
    print("=" * 80)
    
    # Method 1: IQR
    outliers_iqr = detect_outliers_iqr(mean_errors)
    n_iqr = np.sum(outliers_iqr)
    print(f"\n1. IQR Method (k=1.5): {n_iqr} outliers ({100*n_iqr/len(mean_errors):.2f}%)")
    
    # Method 2: Z-score
    outliers_zscore = detect_outliers_zscore(mean_errors)
    n_zscore = np.sum(outliers_zscore)
    print(f"2. Z-score Method (|z|>3): {n_zscore} outliers ({100*n_zscore/len(mean_errors):.2f}%)")
    
    # Method 3: Percentile
    outliers_percentile = detect_outliers_percentile(mean_errors)
    n_percentile = np.sum(outliers_percentile)
    print(f"3. Percentile Method (1st-99th): {n_percentile} outliers ({100*n_percentile/len(mean_errors):.2f}%)")
    
    # Method 4: Isolation Forest
    outliers_if = detect_outliers_isolation_forest(mean_errors)
    n_if = np.sum(outliers_if)
    print(f"4. Isolation Forest (contamination=0.1): {n_if} outliers ({100*n_if/len(mean_errors):.2f}%)")
    
    # Combined (outlier in at least 2 methods)
    outlier_votes = (
        outliers_iqr.astype(int) +
        outliers_zscore.astype(int) +
        outliers_percentile.astype(int) +
        outliers_if.astype(int)
    )
    outliers_combined = outlier_votes >= 2
    n_combined = np.sum(outliers_combined)
    print(f"\n5. Combined (≥2 methods agree): {n_combined} outliers ({100*n_combined/len(mean_errors):.2f}%)")
    
    # Create results dictionary
    results = []
    for i, image_file in enumerate(image_files):
        result = {
            'image_file': image_file,
            'mean_error': float(mean_errors[i]),
            'max_error': float(max_errors[i]),
            'is_outlier_iqr': bool(outliers_iqr[i]),
            'is_outlier_zscore': bool(outliers_zscore[i]),
            'is_outlier_percentile': bool(outliers_percentile[i]),
            'is_outlier_isolation_forest': bool(outliers_if[i]),
            'outlier_votes': int(outlier_votes[i]),
            'is_outlier_combined': bool(outliers_combined[i]),
            'keypoint_errors': keypoint_errors[i].tolist()
        }
        results.append(result)
    
    # Sort by mean error (highest first)
    results.sort(key=lambda x: x['mean_error'], reverse=True)
    
    # Save results
    output_data = {
        'model_used': best_model['config'],
        'model_path': str(model_path),
        'xml_file': str(xml_path),
        'total_samples': len(results),
        'statistics': {
            'mean_error': float(np.mean(mean_errors)),
            'std_error': float(np.std(mean_errors)),
            'median_error': float(np.median(mean_errors)),
            'min_error': float(np.min(mean_errors)),
            'max_error': float(np.max(mean_errors)),
            'q25': float(np.percentile(mean_errors, 25)),
            'q75': float(np.percentile(mean_errors, 75)),
            'iqr': float(np.percentile(mean_errors, 75) - np.percentile(mean_errors, 25))
        },
        'outlier_counts': {
            'iqr': int(n_iqr),
            'zscore': int(n_zscore),
            'percentile': int(n_percentile),
            'isolation_forest': int(n_if),
            'combined': int(n_combined)
        },
        'samples': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Print top outliers
    print("\n" + "=" * 80)
    print("TOP 20 OUTLIERS (by mean error)")
    print("=" * 80)
    for i, result in enumerate(results[:20], 1):
        print(f"\n{i}. {result['image_file']}")
        print(f"   Mean Error: {result['mean_error']:.4f} | Max Error: {result['max_error']:.4f}")
        print(f"   Outlier Methods: IQR={result['is_outlier_iqr']}, Z-score={result['is_outlier_zscore']}, "
              f"Percentile={result['is_outlier_percentile']}, IF={result['is_outlier_isolation_forest']}")
        print(f"   Votes: {result['outlier_votes']}/4")


if __name__ == '__main__':
    main()

