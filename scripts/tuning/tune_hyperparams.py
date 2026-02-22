#!/usr/bin/env python3
"""
Distributed Hyperparameter Tuning with Ray Tune + Bayesian Optimization.

Usage:
    # Single node (1 GPU)
    uv run python scripts/tuning/tune_hyperparams.py --config configs/H5.yaml

    # Multi-node (requires SLURM, see docs/ray_tune_slurm.md)
    uv run python scripts/tuning/tune_hyperparams.py --config configs/H5.yaml --num-samples 50

Requirements:
    pip install ray[tune] optuna ultralytics
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import shutil

import yaml
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune import RunConfig
from ray.tune import Callback
from ray.train import Checkpoint
from ultralytics import YOLO

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Allow large trial artifacts to live on scratch.
DEFAULT_YOLO_PROJECT = str(PROJECT_ROOT / 'runs' / 'tune')
DEFAULT_RAY_STORAGE = str(PROJECT_ROOT / 'runs' / 'ray_results')
YOLO_PROJECT = os.environ.get('YOLO_PROJECT', DEFAULT_YOLO_PROJECT)
RAY_STORAGE_PATH = os.environ.get('RAY_STORAGE_PATH', DEFAULT_RAY_STORAGE)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _resolve_data_yaml(base_config: dict) -> str:
    """Build a data YAML path from the config, matching train_yolo.py logic."""
    dataset_cfg = base_config.get('dataset') if isinstance(base_config.get('dataset'), dict) else None

    # Check train.data first
    train_data = (base_config.get('train') or {}).get('data')
    if train_data and isinstance(train_data, str):
        p = Path(train_data)
        return str(p if p.is_absolute() else PROJECT_ROOT / p)

    # Build from dataset section
    if dataset_cfg:
        data_dict = {}
        for k in ('path', 'train', 'val', 'test', 'nc', 'names'):
            if dataset_cfg.get(k) is not None:
                data_dict[k] = dataset_cfg[k]
        temp_file = PROJECT_ROOT / 'data' / 'tune_data.yaml'
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_file, 'w') as f:
            yaml.dump(data_dict, f, default_flow_style=False)
        return str(temp_file)

    # Fallback
    return str(PROJECT_ROOT / 'data' / 'dataset' / 'data.yaml')


def _resolve_output_paths(base_config: dict) -> tuple[str, str]:
    """Resolve output paths from config first, then environment defaults."""
    tune_cfg = base_config.get('tune', {}) if isinstance(base_config.get('tune'), dict) else {}

    def norm(path_val: str | None) -> str | None:
        if not path_val:
            return None
        expanded = os.path.expandvars(os.path.expanduser(path_val))
        # If env vars are unresolved (e.g., "$FOO"), ignore config value.
        if '$' in expanded:
            return None
        p = Path(expanded)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return str(p)

    yolo_project = norm(tune_cfg.get('project')) or YOLO_PROJECT
    ray_storage = norm(tune_cfg.get('storage_path')) or RAY_STORAGE_PATH
    return yolo_project, ray_storage


def train_yolo_trial(config: dict, base_config: dict = None, yolo_project: str = None):
    """
    Single YOLO training trial for Ray Tune.

    Args:
        config: Hyperparameters from Ray Tune search space
        base_config: Base configuration from YAML file
    """
    # Ensure working directory is project root (Ray workers may start elsewhere)
    os.chdir(PROJECT_ROOT)

    # Merge base config with trial hyperparameters
    train_cfg = base_config.get('train', {}) if base_config else {}

    # Resolve data path from config (same logic as train_yolo.py)
    data_yaml = _resolve_data_yaml(base_config or {})

    # Initialize model with task support
    task = train_cfg.get('task', 'detect')
    model_path = train_cfg.get('model', 'yolo11s.pt')
    if not os.path.isabs(model_path):
        model_path = str(PROJECT_ROOT / model_path)
    model = YOLO(model_path, task=task)

    # Determine workers: respect config, cap to cpus available
    workers = train_cfg.get('workers', 4)

    # Training with trial hyperparameters
    results = model.train(
        data=data_yaml,
        epochs=train_cfg.get('epochs', 100),
        batch=config.get('batch_size', 32),
        imgsz=train_cfg.get('imgsz', 1280),
        workers=workers,
        patience=train_cfg.get('patience', 20),
        cache=train_cfg.get('cache', False),
        lr0=config.get('lr0', 0.01),
        lrf=config.get('lrf', 0.01),
        momentum=config.get('momentum', 0.937),
        weight_decay=config.get('weight_decay', 0.0005),
        warmup_epochs=config.get('warmup_epochs', 3.0),
        box=config.get('box', 7.5),
        cls=config.get('cls', 0.5),
        dfl=config.get('dfl', 1.5),
        hsv_h=config.get('hsv_h', 0.015),
        hsv_s=config.get('hsv_s', 0.7),
        hsv_v=config.get('hsv_v', 0.4),
        degrees=config.get('degrees', 0.0),
        translate=config.get('translate', 0.1),
        scale=config.get('scale', 0.5),
        fliplr=config.get('fliplr', 0.5),
        mosaic=config.get('mosaic', 1.0),
        mixup=config.get('mixup', 0.0),
        cos_lr=train_cfg.get('cos_lr', False),
        close_mosaic=train_cfg.get('close_mosaic', 10),
        project=yolo_project or YOLO_PROJECT,
        name=f"trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        exist_ok=True,
        verbose=False,
    )

    # YOLO's built-in callback reports per-epoch metrics to Ray Tune.
    # After training, save the best weights as a Ray Checkpoint so that
    # results.get_best_result().checkpoint can retrieve the best model.
    weights_dir = str(Path(results.save_dir) / "weights")
    checkpoint = Checkpoint.from_directory(weights_dir)
    ray.train.report(
        {
            "metrics/mAP50(B)": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "metrics/mAP50-95(B)": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "metrics/precision(B)": float(results.results_dict.get("metrics/precision(B)", 0)),
            "metrics/recall(B)": float(results.results_dict.get("metrics/recall(B)", 0)),
            "yolo_save_dir": str(results.save_dir),
        },
        checkpoint=checkpoint,
    )


def get_search_space(tune_config: dict = None):
    """
    Define hyperparameter search space.

    Args:
        tune_config: Optional config to override defaults
    """
    # Default search space for YOLO
    space = {
        # Learning rate
        'lr0': tune.loguniform(1e-5, 1e-1),
        'lrf': tune.uniform(0.01, 0.5),

        # Optimizer
        'momentum': tune.uniform(0.8, 0.98),
        'weight_decay': tune.loguniform(1e-5, 1e-2),
        'warmup_epochs': tune.uniform(1.0, 5.0),

        # Batch size
        'batch_size': tune.choice([4, 8, 16]),

        # Loss weights
        'box': tune.uniform(5.0, 10.0),
        'cls': tune.uniform(0.3, 1.0),
        'dfl': tune.uniform(1.0, 2.0),

        # Augmentation
        'hsv_h': tune.uniform(0.0, 0.1),
        'hsv_s': tune.uniform(0.0, 0.9),
        'hsv_v': tune.uniform(0.0, 0.9),
        'degrees': tune.uniform(0.0, 45.0),
        'translate': tune.uniform(0.0, 0.5),
        'scale': tune.uniform(0.0, 0.9),
        'fliplr': tune.uniform(0.0, 1.0),
        'mosaic': tune.uniform(0.0, 1.0),
        'mixup': tune.uniform(0.0, 0.5),
    }

    # Override with user config if provided
    if tune_config:
        for key, value in tune_config.get('search_space', {}).items():
            if isinstance(value, dict):
                if value.get('type') == 'loguniform':
                    space[key] = tune.loguniform(value['min'], value['max'])
                elif value.get('type') == 'uniform':
                    space[key] = tune.uniform(value['min'], value['max'])
                elif value.get('type') == 'choice':
                    space[key] = tune.choice(value['values'])

    return space


class BestTrialCallback(Callback):
    """Save best hyperparams + model after each trial completes."""

    def __init__(self, metric: str, mode: str):
        self.metric = metric
        self.mode = mode
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_config_path = PROJECT_ROOT / 'configs' / 'best_hyperparams.yaml'
        self.best_model_dir = PROJECT_ROOT / 'models' / 'best_tune'

    def on_trial_complete(self, iteration, trials, trial, **info):
        value = trial.last_result.get(self.metric)
        if value is None:
            return

        improved = (self.mode == 'max' and value > self.best_value) or \
                   (self.mode == 'min' and value < self.best_value)
        if not improved:
            return

        self.best_value = value
        metrics = trial.last_result
        yolo_save_dir = metrics.get("yolo_save_dir")

        print(f"\n*** New best trial! {self.metric}={value:.4f} ***")

        # Save best hyperparams yaml
        with open(self.best_config_path, 'w') as f:
            yaml.dump({
                'tune_results': {
                    'best_config': trial.config,
                    'best_metrics': {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                    'yolo_save_dir': yolo_save_dir,
                    'ray_trial_dir': trial.path,
                    'timestamp': datetime.now().isoformat(),
                }
            }, f, default_flow_style=False)
        print(f"  Saved config → {self.best_config_path}")

        # Copy best model weights
        if yolo_save_dir:
            src_weights = Path(yolo_save_dir) / "weights" / "best.pt"
            if src_weights.exists():
                self.best_model_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_weights, self.best_model_dir / "best.pt")
                print(f"  Saved model  → {self.best_model_dir / 'best.pt'}")


def run_tuning(
    config_path: str,
    num_samples: int = 20,
    max_concurrent: int = None,
    gpus_per_trial: float = 1.0,
    cpus_per_trial: int = 4,
    metric: str = 'metrics/mAP50(B)',
    mode: str = 'max',
    resume: bool = False,
):
    """
    Run distributed hyperparameter tuning.

    Args:
        config_path: Path to YAML config file
        num_samples: Number of hyperparameter combinations to try
        max_concurrent: Max concurrent trials (None = auto)
        gpus_per_trial: GPUs per trial
        cpus_per_trial: CPUs per trial
        metric: Metric to optimize
        mode: 'max' or 'min'
        resume: Resume from previous run
    """
    # Load base config
    base_config = load_config(config_path) if config_path else {}
    tune_config = base_config.get('tune', {})
    yolo_project, ray_storage_path = _resolve_output_paths(base_config)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    print(f"Ray initialized with {ray.cluster_resources()}")

    # Search space
    search_space = get_search_space(tune_config)

    # Extract seed point from train config (e.g. H6 tuned params) if available
    train_cfg = base_config.get('train', {}) if base_config else {}
    seed_keys = [
        'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs',
        'batch_size', 'box', 'cls', 'dfl',
        'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
        'fliplr', 'mosaic', 'mixup',
    ]
    # Map config key 'batch' to search space key 'batch_size'
    seed_point = {}
    for k in seed_keys:
        cfg_key = 'batch' if k == 'batch_size' else k
        if cfg_key in train_cfg:
            seed_point[k] = train_cfg[cfg_key]

    points_to_evaluate = [seed_point] if seed_point else None
    if points_to_evaluate:
        print(f"Seeding Optuna with {len(seed_point)} params from config")

    # Bayesian search with Optuna
    search_alg = OptunaSearch(
        metric=metric,
        mode=mode,
        points_to_evaluate=points_to_evaluate,
    )

    # Early stopping scheduler (ASHA)
    # ASHA rungs with reduction_factor=3: 30 → 90 → 270 → 300
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric=metric,
        mode=mode,
        max_t=300,  # max epochs (full training)
        grace_period=20,  # min epochs before stopping
        reduction_factor=3,
    )

    # Trainable with base config
    trainable = tune.with_parameters(
        train_yolo_trial,
        base_config=base_config,
        yolo_project=yolo_project,
    )
    trainable = tune.with_resources(trainable, {"cpu": cpus_per_trial, "gpu": gpus_per_trial})

    # Run tuning
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
        ),
        run_config=RunConfig(
            name=f"yolo_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            storage_path=ray_storage_path,
            verbose=1,
            callbacks=[BestTrialCallback(metric=metric, mode=mode)],
        ),
    )

    if resume:
        tuner = tune.Tuner.restore(
            ray_storage_path,
            trainable=trainable,
        )

    results = tuner.fit()

    # Print best results
    best_result = results.get_best_result(metric=metric, mode=mode)
    best_metrics = best_result.metrics
    yolo_save_dir = best_metrics.get("yolo_save_dir")
    ray_trial_dir = best_result.path

    print("\n" + "=" * 60)
    print("TUNING COMPLETE — BEST TRIAL SUMMARY")
    print("=" * 60)

    print("\n--- Metrics ---")
    print(f"  mAP50:     {best_metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP50-95:  {best_metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"  Precision: {best_metrics.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"  Recall:    {best_metrics.get('metrics/recall(B)', 'N/A'):.4f}")

    print("\n--- Hyperparameters ---")
    for key, value in best_result.config.items():
        print(f"  {key}: {value}")

    print("\n--- Result Paths ---")
    print(f"  Ray trial dir:   {ray_trial_dir}")
    print(f"    progress.csv   — per-epoch metrics")
    print(f"    result.json    — final metrics")
    print(f"    params.json    — hyperparameters")
    if yolo_save_dir:
        print(f"  YOLO output dir: {yolo_save_dir}")
        print(f"    weights/best.pt — best model weights")
        print(f"    results.csv     — YOLO training log")

    # Save best model checkpoint to project directory
    best_model_dir = PROJECT_ROOT / 'models' / 'best_tune'
    if best_result.checkpoint:
        best_result.checkpoint.to_directory(str(best_model_dir))
        print(f"  Copied best.pt → {best_model_dir / 'best.pt'}")

    print("=" * 60)

    # Save best config
    best_config_path = PROJECT_ROOT / 'configs' / 'best_hyperparams.yaml'
    with open(best_config_path, 'w') as f:
        yaml.dump({
            'tune_results': {
                'best_config': best_result.config,
                'best_metrics': {k: float(v) for k, v in best_metrics.items() if isinstance(v, (int, float))},
                'best_model_path': str(best_model_dir / 'best.pt') if best_result.checkpoint else None,
                'yolo_save_dir': yolo_save_dir,
                'ray_trial_dir': ray_trial_dir,
                'timestamp': datetime.now().isoformat(),
            }
        }, f, default_flow_style=False)
    print(f"\nAll results saved to: {best_config_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Distributed Hyperparameter Tuning')
    parser.add_argument('--config', type=str, default='configs/H5.yaml',
                        help='Path to config file')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of hyperparameter combinations to try')
    parser.add_argument('--max-concurrent', type=int, default=None,
                        help='Max concurrent trials (default: auto based on resources)')
    parser.add_argument('--gpus-per-trial', type=float, default=1.0,
                        help='GPUs per trial')
    parser.add_argument('--cpus-per-trial', type=int, default=4,
                        help='CPUs per trial')
    parser.add_argument('--metric', type=str, default='metrics/mAP50(B)',
                        choices=[
                            'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                            'metrics/precision(B)', 'metrics/recall(B)',
                        ],
                        help='Metric to optimize (use YOLO native metric names)')
    parser.add_argument('--mode', type=str, default='max',
                        choices=['max', 'min'],
                        help='Optimization mode')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run')

    args = parser.parse_args()

    run_tuning(
        config_path=args.config,
        num_samples=args.num_samples,
        max_concurrent=args.max_concurrent,
        gpus_per_trial=args.gpus_per_trial,
        cpus_per_trial=args.cpus_per_trial,
        metric=args.metric,
        mode=args.mode,
        resume=args.resume,
    )


if __name__ == '__main__':
    main()
