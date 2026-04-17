#!/usr/bin/env python3
"""
Log training results to MLflow and register model versions.

Counterpart to publish_release.py — while that script publishes to GitHub
Releases, this one logs metrics/params to MLflow Tracking and registers the
model in the MLflow Model Registry.

Usage:
    # Dry run
    python scripts/deployment/register_model.py \
        --config configs/H11_obb.yaml --version v1.0.0-obb --dry-run

    # Log to MLflow
    python scripts/deployment/register_model.py \
        --config configs/H11_obb.yaml --version v1.0.0-obb \
        --tracking-uri http://localhost:5000
"""

import argparse
import csv
import sys
from pathlib import Path

import mlflow
import yaml


def get_best_metrics(results_csv: Path) -> dict:
    """Read best epoch from results.csv (by mAP50-95)."""
    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    # Strip whitespace from column names (YOLO adds trailing spaces)
    rows = [{k.strip(): v.strip() for k, v in row.items()} for row in rows]
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0)))
    total_epochs = int(float(rows[-1].get("epoch", 0))) + 1
    best_epoch = int(float(best.get("epoch", 0))) + 1
    return {
        "best_epoch": best_epoch,
        "epochs_completed": total_epochs,
        "mAP50": round(float(best.get("metrics/mAP50(B)", 0)), 5),
        "mAP50-95": round(float(best.get("metrics/mAP50-95(B)", 0)), 5),
        "precision": round(float(best.get("metrics/precision(B)", 0)), 5),
        "recall": round(float(best.get("metrics/recall(B)", 0)), 5),
    }


def main():
    parser = argparse.ArgumentParser(description="Log training results to MLflow and register model")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--version", required=True, help="Version tag (e.g. v1.0.0-obb)")
    parser.add_argument("--tracking-uri", default="http://localhost:5000", help="MLflow tracking server URL")
    parser.add_argument("--experiment-name", default=None, help="MLflow experiment name (defaults to config name)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be logged")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]
    task = train_cfg.get("task", "detect")
    name = train_cfg["name"]
    run_dir = Path(f"runs/{task}/{name}")

    if not run_dir.exists():
        print(f"Error: run directory not found: {run_dir}")
        sys.exit(1)

    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        print(f"Error: results.csv not found: {results_csv}")
        sys.exit(1)

    weights_path = run_dir / "weights" / "best.pt"
    if not weights_path.exists():
        print(f"Error: best.pt not found: {weights_path}")
        sys.exit(1)

    # Extract metrics
    metrics = get_best_metrics(results_csv)
    if not metrics:
        print("Error: no metrics found in results.csv")
        sys.exit(1)

    # Training params to log
    params = {
        "epochs": train_cfg.get("epochs"),
        "batch": train_cfg.get("batch"),
        "imgsz": train_cfg.get("imgsz"),
        "patience": train_cfg.get("patience"),
        "model": train_cfg.get("model"),
        "task": task,
    }

    experiment_name = args.experiment_name or name
    run_name = f"{name}_{args.version}"

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Config:          {name}")
    print(f"Version:         {args.version}")
    print(f"Tracking URI:    {args.tracking_uri}")
    print(f"Experiment:      {experiment_name}")
    print(f"Run name:        {run_name}")
    print(f"Weights:         {weights_path}")
    print(f"\nParams:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nTag: github_release_tag = model/{args.version}")

    if args.dry_run:
        print(f"\n[DRY RUN] — nothing logged to MLflow")
        return

    # Log to MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("github_release_tag", f"model/{args.version}")
        run_id = run.info.run_id

    # Register in Model Registry — source points to GitHub Release, no artifact upload
    client = mlflow.MlflowClient()
    try:
        client.create_registered_model(name)
    except mlflow.exceptions.MlflowException:
        pass  # already exists

    release_url = f"https://github.com/Human-Augment-Analytics/Lizard_Toepads/releases/tag/model/{args.version}"
    mv = client.create_model_version(name=name, source=release_url, run_id=run_id)

    print(f"\nLogged to MLflow:")
    print(f"  Run ID:        {run_id}")
    print(f"  Experiment:    {experiment_name}")
    print(f"  Model:         {name} version {mv.version}")
    print(f"  Source:        {release_url}")
    print(f"  Tracking URI:  {args.tracking_uri}")

    print("\nDone.")


if __name__ == "__main__":
    main()
