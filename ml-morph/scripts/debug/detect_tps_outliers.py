#!/usr/bin/env python3
"""
Detect outliers in TPS landmark files based on statistical analysis of coordinates.
"""
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except (ImportError, AttributeError):
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available. Isolation Forest method will be disabled.")


def read_tps(input_file: str) -> Dict:
    """
    Read TPS file and return parsed data.
    
    Args:
        input_file: Path to TPS file
        
    Returns:
        Dictionary with keys: 'lm', 'im', 'scl', 'coords'
    """
    with open(input_file, 'r') as f:
        tps = f.read().splitlines()
    
    lm, im, sc, coords_array = [], [], [], []
    
    for i, ln in enumerate(tps):
        if ln.startswith("LM"):
            lm_num = int(ln.split('=')[1])
            lm.append(lm_num)
            coords_mat = []
            for j in range(i + 1, i + 1 + lm_num):
                if j < len(tps) and tps[j].strip():
                    parts = tps[j].split()
                    if len(parts) >= 2:
                        coords_mat.append([float(parts[0]), float(parts[1])])
            if coords_mat:
                coords_array.append(np.array(coords_mat, dtype=float))
        
        if ln.startswith("IMAGE"):
            im.append(ln.split('=')[1])
        
        if ln.startswith("SCALE"):
            sc.append(ln.split('=')[1])
    
    return {'lm': lm, 'im': im, 'scl': sc, 'coords': coords_array}


def compute_image_statistics(coords: np.ndarray) -> Dict[str, float]:
    """
    Compute various statistics for a single image's landmarks.
    
    Args:
        coords: Nx2 array of landmark coordinates
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Basic bounding box
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    stats['bbox_width'] = max_x - min_x
    stats['bbox_height'] = max_y - min_y
    stats['bbox_area'] = stats['bbox_width'] * stats['bbox_height']
    stats['bbox_aspect_ratio'] = stats['bbox_width'] / stats['bbox_height'] if stats['bbox_height'] > 0 else 0
    
    # Center of landmarks
    center_x = coords[:, 0].mean()
    center_y = coords[:, 1].mean()
    stats['center_x'] = center_x
    stats['center_y'] = center_y
    
    # Distances from center
    distances_from_center = np.sqrt((coords[:, 0] - center_x)**2 + (coords[:, 1] - center_y)**2)
    stats['mean_distance_from_center'] = distances_from_center.mean()
    stats['max_distance_from_center'] = distances_from_center.max()
    stats['std_distance_from_center'] = distances_from_center.std()
    
    # Pairwise distances between consecutive landmarks
    if len(coords) > 1:
        pairwise_dists = []
        for i in range(len(coords) - 1):
            dist = np.linalg.norm(coords[i+1] - coords[i])
            pairwise_dists.append(dist)
        
        if pairwise_dists:
            stats['mean_pairwise_distance'] = np.mean(pairwise_dists)
            stats['std_pairwise_distance'] = np.std(pairwise_dists)
            stats['min_pairwise_distance'] = np.min(pairwise_dists)
            stats['max_pairwise_distance'] = np.max(pairwise_dists)
    
    # Total path length (sum of consecutive distances)
    if len(coords) > 1:
        path_length = 0
        for i in range(len(coords) - 1):
            path_length += np.linalg.norm(coords[i+1] - coords[i])
        stats['total_path_length'] = path_length
    
    # Spread metrics
    stats['x_range'] = coords[:, 0].max() - coords[:, 0].min()
    stats['y_range'] = coords[:, 1].max() - coords[:, 1].min()
    stats['x_std'] = coords[:, 0].std()
    stats['y_std'] = coords[:, 1].std()
    
    # Number of landmarks
    stats['num_landmarks'] = len(coords)
    
    return stats


def detect_outliers_iqr(values: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Detect outliers using IQR method."""
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    if IQR == 0:
        return np.zeros(len(values), dtype=bool)
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return (values < lower_bound) | (values > upper_bound)


def detect_outliers_zscore(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Detect outliers using Z-score method."""
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return np.zeros(len(values), dtype=bool)
    z_scores = np.abs((values - mean) / std)
    return z_scores > threshold


def detect_outliers_percentile(values: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """Detect outliers using percentile method."""
    lower_bound = np.percentile(values, lower)
    upper_bound = np.percentile(values, upper)
    return (values < lower_bound) | (values > upper_bound)


def detect_outliers_isolation_forest(features: np.ndarray, contamination: float = 0.1) -> np.ndarray:
    """Detect outliers using Isolation Forest."""
    if not HAS_SKLEARN:
        return np.zeros(len(features), dtype=bool)
    
    # Reshape if 1D
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(features)
    return outliers == -1


def main():
    parser = argparse.ArgumentParser(
        description='Detect outliers in TPS landmark files'
    )
    parser.add_argument(
        '--tps-file',
        type=str,
        default='ml-morph/consolidated_finger.tps',
        help='Path to TPS file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ml-morph/tps_outliers_results.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--output-report',
        type=str,
        default='ml-morph/tps_outliers_report.txt',
        help='Output text report file path'
    )
    
    args = parser.parse_args()
    
    print(f"Reading TPS file: {args.tps_file}")
    tps_data = read_tps(args.tps_file)
    
    n_samples = len(tps_data['coords'])
    print(f"Found {n_samples} samples")
    
    if n_samples == 0:
        print("Error: No samples found in TPS file!")
        return
    
    # Compute statistics for each image
    print("\nComputing statistics for each image...")
    all_stats = []
    image_names = tps_data['im'] if tps_data['im'] else [f"image_{i}" for i in range(n_samples)]
    
    for i, coords in enumerate(tps_data['coords']):
        stats = compute_image_statistics(coords)
        stats['image_name'] = image_names[i] if i < len(image_names) else f"image_{i}"
        stats['index'] = i
        all_stats.append(stats)
    
    # Extract feature arrays for outlier detection
    feature_keys = [
        'bbox_width', 'bbox_height', 'bbox_area', 'bbox_aspect_ratio',
        'mean_distance_from_center', 'max_distance_from_center',
        'mean_pairwise_distance', 'std_pairwise_distance',
        'total_path_length', 'x_range', 'y_range', 'x_std', 'y_std'
    ]
    
    # Filter out None values and create feature matrix
    features_dict = {}
    for key in feature_keys:
        values = [s.get(key) for s in all_stats if s.get(key) is not None]
        if values:
            features_dict[key] = np.array(values)
    
    # Detect outliers for each feature
    print("\nDetecting outliers...")
    outlier_flags = {}
    
    for key, values in features_dict.items():
        # IQR
        outliers_iqr = detect_outliers_iqr(values)
        # Z-score
        outliers_zscore = detect_outliers_zscore(values)
        # Percentile
        outliers_percentile = detect_outliers_percentile(values)
        
        outlier_flags[key] = {
            'iqr': outliers_iqr.tolist(),
            'zscore': outliers_zscore.tolist(),
            'percentile': outliers_percentile.tolist()
        }
    
    # Multivariate outlier detection using Isolation Forest on all features
    if features_dict:
        # Combine all features into a matrix
        feature_matrix = np.column_stack([features_dict[key] for key in feature_keys if key in features_dict])
        outliers_multivariate = detect_outliers_isolation_forest(feature_matrix)
        outlier_flags['multivariate'] = {
            'isolation_forest': outliers_multivariate.tolist()
        }
    
    # Aggregate outlier votes per image
    print("\nAggregating outlier votes...")
    for i, stats in enumerate(all_stats):
        votes = 0
        methods_flagged = []
        
        # Check each feature-based method
        for key, flags in outlier_flags.items():
            if key == 'multivariate':
                continue
            for method, flag_list in flags.items():
                if i < len(flag_list) and flag_list[i]:
                    votes += 1
                    methods_flagged.append(f"{key}_{method}")
        
        # Check multivariate method
        if 'multivariate' in outlier_flags:
            if i < len(outlier_flags['multivariate']['isolation_forest']):
                if outlier_flags['multivariate']['isolation_forest'][i]:
                    votes += 2  # Weight multivariate higher
                    methods_flagged.append("multivariate_isolation_forest")
        
        stats['outlier_votes'] = votes
        stats['outlier_methods'] = methods_flagged
        stats['is_outlier'] = votes >= 2  # Flag if 2+ methods agree (lowered threshold)
    
    # Sort by outlier votes
    all_stats.sort(key=lambda x: x['outlier_votes'], reverse=True)
    
    # Print summary
    print("\n" + "=" * 80)
    print("OUTLIER DETECTION SUMMARY")
    print("=" * 80)
    
    n_outliers = sum(1 for s in all_stats if s['is_outlier'])
    print(f"\nTotal samples: {n_samples}")
    print(f"Outliers detected (>=2 votes): {n_outliers} ({100*n_outliers/n_samples:.2f}%)")
    
    # Show top outliers
    print("\nTop 20 potential outliers:")
    print("-" * 80)
    print(f"{'Image':<25} {'Votes':<8} {'Methods':<30} {'BBox Area':<12}")
    print("-" * 80)
    
    for stats in all_stats[:20]:
        if stats['outlier_votes'] > 0:
            methods_str = ', '.join(stats['outlier_methods'][:3])
            if len(stats['outlier_methods']) > 3:
                methods_str += "..."
            bbox_area = stats.get('bbox_area', 0)
            print(f"{stats['image_name']:<25} {stats['outlier_votes']:<8} {methods_str:<30} {bbox_area:>12.2f}")
    
    # Save results
    output_data = {
        'summary': {
            'total_samples': n_samples,
            'outliers_detected': n_outliers,
            'outlier_percentage': 100 * n_outliers / n_samples
        },
        'statistics': all_stats,
        'outlier_flags': outlier_flags
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Write text report
    with open(args.output_report, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TPS OUTLIER DETECTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"TPS File: {args.tps_file}\n")
        f.write(f"Total Samples: {n_samples}\n")
        f.write(f"Outliers Detected (>=2 votes): {n_outliers} ({100*n_outliers/n_samples:.2f}%)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("OUTLIER DETAILS\n")
        f.write("=" * 80 + "\n\n")
        
        for stats in all_stats:
            if stats['outlier_votes'] > 0:
                f.write(f"Image: {stats['image_name']}\n")
                f.write(f"  Index: {stats['index']}\n")
                f.write(f"  Outlier Votes: {stats['outlier_votes']}\n")
                f.write(f"  Methods Flagged: {', '.join(stats['outlier_methods'])}\n")
                f.write(f"  Statistics:\n")
                for key in ['bbox_width', 'bbox_height', 'bbox_area', 'num_landmarks']:
                    if key in stats:
                        f.write(f"    {key}: {stats[key]:.2f}\n")
                f.write("\n")
    
    print(f"Text report saved to: {args.output_report}")


if __name__ == '__main__':
    main()

