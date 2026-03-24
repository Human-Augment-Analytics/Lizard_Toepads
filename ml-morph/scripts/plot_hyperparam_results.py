#!/usr/bin/env python3
"""
Generate visualization plots for hyperparameter search results.
Saves plots as PNG files to the results directory.
"""
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir):
    results_file = os.path.join(results_dir, 'results.json')
    with open(results_file) as f:
        return json.load(f)


def plot_all(results, output_dir, category='toe'):
    """Generate all plots and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    train_errors = [r['train_error'] for r in results]
    test_errors = [r['test_error'] for r in results]
    overfitting = [r['overfitting'] for r in results]

    # Extract parameter values
    tree_depths = [r['params']['tree_depth'] for r in results]
    cascade_depths = [r['params']['cascade_depth'] for r in results]
    nus = [r['params']['nu'] for r in results]
    num_trees = [r['params']['num_trees'] for r in results]
    oversampling = [r['params']['oversampling'] for r in results]

    # --- Figure 1: Train vs Test Error scatter ---
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(train_errors, test_errors, c=overfitting, cmap='RdYlGn_r',
                    alpha=0.7, edgecolors='k', linewidths=0.5, s=60)
    plt.colorbar(sc, ax=ax, label='Overfitting (test - train)')
    ax.set_xlabel('Train Error')
    ax.set_ylabel('Test Error')
    ax.set_title(f'Train vs Test Error — {category.upper()} Hyperparam Search ({len(results)} configs)')
    # Mark best model
    best_idx = np.argmin(test_errors)
    ax.scatter([train_errors[best_idx]], [test_errors[best_idx]],
               marker='*', s=300, c='red', zorder=5, label=f'Best (test={test_errors[best_idx]:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{category}_train_vs_test.png'), dpi=150)
    plt.close(fig)

    # --- Figure 2: Test error by each hyperparameter ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Test Error by Hyperparameter — {category.upper()}', fontsize=14)

    param_pairs = [
        ('tree_depth', tree_depths),
        ('cascade_depth', cascade_depths),
        ('nu', nus),
        ('num_trees', num_trees),
        ('oversampling', oversampling),
    ]

    for idx, (name, vals) in enumerate(param_pairs):
        ax = axes.flat[idx]
        unique_vals = sorted(set(vals))
        grouped = {v: [] for v in unique_vals}
        for v, te in zip(vals, test_errors):
            grouped[v].append(te)

        positions = range(len(unique_vals))
        bp = ax.boxplot([grouped[v] for v in unique_vals], positions=positions,
                        patch_artist=True, widths=0.6)
        for patch in bp['boxes']:
            patch.set_facecolor('#4C9ED9')
            patch.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(v) for v in unique_vals])
        ax.set_xlabel(name)
        ax.set_ylabel('Test Error')
        ax.set_title(name)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplot
    axes.flat[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{category}_test_error_by_param.png'), dpi=150)
    plt.close(fig)

    # --- Figure 3: Top 10 models bar chart ---
    sorted_results = sorted(results, key=lambda x: x['test_error'])
    top10 = sorted_results[:10]

    fig, ax = plt.subplots(figsize=(14, 7))
    labels = []
    for r in top10:
        p = r['params']
        labels.append(f"d{p['tree_depth']}_c{p['cascade_depth']}_nu{p['nu']}_t{p['num_trees']}_o{p['oversampling']}")

    x = np.arange(len(top10))
    width = 0.35
    bars_train = ax.bar(x - width/2, [r['train_error'] for r in top10], width, label='Train Error', color='#4C9ED9')
    bars_test = ax.bar(x + width/2, [r['test_error'] for r in top10], width, label='Test Error', color='#E74C3C')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Error')
    ax.set_title(f'Top 10 Models by Test Error — {category.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{category}_top10_models.png'), dpi=150)
    plt.close(fig)

    # --- Figure 4: Heatmap — tree_depth vs cascade_depth (mean test error) ---
    unique_td = sorted(set(tree_depths))
    unique_cd = sorted(set(cascade_depths))
    heatmap = np.full((len(unique_td), len(unique_cd)), np.nan)

    for r in results:
        td_idx = unique_td.index(r['params']['tree_depth'])
        cd_idx = unique_cd.index(r['params']['cascade_depth'])
        if np.isnan(heatmap[td_idx, cd_idx]):
            heatmap[td_idx, cd_idx] = r['test_error']
        else:
            # Average over other params
            heatmap[td_idx, cd_idx] = (heatmap[td_idx, cd_idx] + r['test_error']) / 2

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap, cmap='RdYlGn_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='Mean Test Error')
    ax.set_xticks(range(len(unique_cd)))
    ax.set_xticklabels(unique_cd)
    ax.set_yticks(range(len(unique_td)))
    ax.set_yticklabels(unique_td)
    ax.set_xlabel('cascade_depth')
    ax.set_ylabel('tree_depth')
    ax.set_title(f'Mean Test Error: tree_depth vs cascade_depth — {category.upper()}')
    # Annotate cells
    for i in range(len(unique_td)):
        for j in range(len(unique_cd)):
            if not np.isnan(heatmap[i, j]):
                ax.text(j, i, f'{heatmap[i, j]:.1f}', ha='center', va='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{category}_heatmap_depth_cascade.png'), dpi=150)
    plt.close(fig)

    print(f"Saved 4 plots to {output_dir}/")
    print(f"  - {category}_train_vs_test.png")
    print(f"  - {category}_test_error_by_param.png")
    print(f"  - {category}_top10_models.png")
    print(f"  - {category}_heatmap_depth_cascade.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot hyperparameter search results')
    parser.add_argument('--results-dir', required=True, help='Directory containing results.json')
    parser.add_argument('--category', default='toe', help='Category name (toe/finger)')
    args = parser.parse_args()

    results = load_results(args.results_dir)
    plot_all(results, args.results_dir, args.category)


if __name__ == '__main__':
    main()
