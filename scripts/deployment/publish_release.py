#!/usr/bin/env python3
"""
Publish trained models to GitHub Releases with semantic versioning.

Uses GitHub REST API (requests) — no gh CLI needed.

Versioning scheme:
    v{MAJOR}.{MINOR}.{PATCH}-{task}
    - MAJOR: architecture change (OBB vs detect)
    - MINOR: training improvement (hyperparams, data, retraining)
    - PATCH: export/quantization change

Assets per release:
    best.pt, best_fp16.onnx, best_fp32.onnx, metadata.json

Setup:
    export GITHUB_TOKEN=ghp_xxxx   # GitHub personal access token (repo scope)

Usage:
    # Dry run
    python scripts/deployment/publish_release.py \
        --config configs/H11_obb.yaml --version v1.0.0-obb --dry-run

    # Publish
    python scripts/deployment/publish_release.py \
        --config configs/H11_obb.yaml --version v1.0.0-obb
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import subprocess

import requests
import yaml


REPO = "Human-Augment-Analytics/Lizard_Toepads"
API_BASE = "https://api.github.com"


def load_dotenv():
    """Load .env file from project root if it exists."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def get_token() -> str:
    """Get GitHub token from .env or environment."""
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set")
        print("  export GITHUB_TOKEN=ghp_xxxx")
        print("  Create one at: https://github.com/settings/tokens (repo scope)")
        sys.exit(1)
    return token


def create_release(repo: str, tag: str, title: str, notes: str, token: str) -> dict:
    """Create a GitHub Release via REST API. Returns release JSON."""
    url = f"{API_BASE}/repos/{repo}/releases"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        },
        json={
            "tag_name": tag,
            "name": title,
            "body": notes,
            "draft": False,
            "prerelease": False,
        },
    )
    if resp.status_code != 201:
        print(f"Error creating release: {resp.status_code}")
        print(resp.json().get("message", resp.text))
        sys.exit(1)
    return resp.json()


def upload_asset(upload_url: str, file_path: Path, token: str) -> None:
    """Upload a single asset to a GitHub Release."""
    # upload_url has {?name,label} template — strip it
    upload_url = upload_url.split("{")[0]
    name = file_path.name
    size = file_path.stat().st_size

    # Set content type
    if name.endswith(".onnx"):
        content_type = "application/octet-stream"
    elif name.endswith(".pt"):
        content_type = "application/octet-stream"
    elif name.endswith(".json"):
        content_type = "application/json"
    else:
        content_type = "application/octet-stream"

    print(f"  Uploading {name} ({size / 1e6:.1f} MB)...", end=" ", flush=True)
    with open(file_path, "rb") as f:
        resp = requests.post(
            upload_url,
            headers={
                "Authorization": f"token {token}",
                "Content-Type": content_type,
            },
            params={"name": name},
            data=f,
        )
    if resp.status_code == 201:
        print("OK")
    else:
        print(f"FAILED ({resp.status_code}: {resp.json().get('message', '')})")


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


def build_metadata(cfg: dict, run_dir: Path, version: str) -> dict:
    """Build metadata.json content from config and training results."""
    train_cfg = cfg.get("train", {})
    dataset_cfg = cfg.get("dataset", {})
    results_csv = run_dir / "results.csv"

    metrics = get_best_metrics(results_csv) if results_csv.exists() else {}

    # Count dataset images
    dataset_path = Path(dataset_cfg.get("path", ""))
    train_rel = dataset_cfg.get("train", "images/train")
    val_rel = dataset_cfg.get("val", "images/val")
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def count_images(directory: Path) -> int:
        if not directory.exists():
            return 0
        return sum(1 for f in directory.iterdir() if f.suffix.lower() in image_exts)

    n_train = count_images(dataset_path / train_rel)
    n_val = count_images(dataset_path / val_rel)

    # Get author from git config
    try:
        git_name = subprocess.run(
            ["git", "config", "user.name"], capture_output=True, text=True
        ).stdout.strip()
        git_email = subprocess.run(
            ["git", "config", "user.email"], capture_output=True, text=True
        ).stdout.strip()
        author = f"{git_name} <{git_email}>" if git_name else "unknown"
    except Exception:
        author = "unknown"

    return {
        "version": version,
        "trained": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "author": author,
        "architecture": train_cfg.get("model", "unknown"),
        "task": train_cfg.get("task", "detect"),
        "config": train_cfg.get("name", "unknown"),
        "dataset": {
            "nc": dataset_cfg.get("nc"),
            "names": dataset_cfg.get("names", []),
            "path": dataset_cfg.get("path"),
            "train_images": n_train,
            "val_images": n_val,
        },
        "training": {
            "epochs": train_cfg.get("epochs"),
            "batch": train_cfg.get("batch"),
            "imgsz": train_cfg.get("imgsz"),
            "patience": train_cfg.get("patience"),
        },
        "metrics": metrics,
        "assets": {
            "pt": "best.pt",
            "fp16": "best_fp16.onnx",
            "fp32": "best_fp32.onnx",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Publish model to GitHub Releases")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--version", required=True, help="Version tag (e.g. v1.0.0-obb)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be published")
    parser.add_argument("--repo", default=REPO, help="GitHub repo (owner/name)")
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

    weights_dir = run_dir / "weights"
    pt = weights_dir / "best.pt"
    fp16 = weights_dir / "best_fp16.onnx"
    fp32 = weights_dir / "best_fp32.onnx"

    # Check assets exist
    assets = []
    for filepath in [pt, fp16, fp32]:
        if filepath.exists():
            assets.append(filepath)
            print(f"  Found: {filepath} ({filepath.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  Missing: {filepath}")

    if not assets:
        print("Error: no model files found")
        sys.exit(1)

    # Build and write metadata.json
    metadata = build_metadata(cfg, run_dir, args.version)
    metadata_path = weights_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    assets.append(metadata_path)
    print(f"  Created: {metadata_path}")

    # Build release notes
    m = metadata["metrics"]
    notes = (
        f"## {name} ({args.version})\n\n"
        f"- **author**: {metadata['author']}\n"
        f"- **trained**: {metadata['trained']}\n"
        f"- **architecture**: {metadata['architecture']}\n"
        f"- **task**: {task}\n"
        f"- **dataset**: {metadata['dataset']['nc']} classes — {metadata['dataset']['names']}\n"
        f"- **images**: {metadata['dataset']['train_images']} train / {metadata['dataset']['val_images']} val\n"
        f"- **best_epoch**: {m.get('best_epoch', '?')}/{m.get('epochs_completed', '?')} (patience {train_cfg.get('patience')})\n"
        f"- **mAP50**: {m.get('mAP50', '?')}\n"
        f"- **mAP50-95**: {m.get('mAP50-95', '?')}\n"
        f"- **precision**: {m.get('precision', '?')}\n"
        f"- **recall**: {m.get('recall', '?')}\n"
    )

    print(f"\n{'=' * 50}")
    print(f"Tag: model/{args.version}")
    print(f"Repo: {args.repo}")
    print(f"Assets: {[str(a) for a in assets]}")
    print(f"\nRelease notes:\n{notes}")

    if args.dry_run:
        print("[DRY RUN] — no release created")
        return

    # Publish via GitHub API
    token = get_token()

    print("\nCreating release...")
    release = create_release(
        repo=args.repo,
        tag=f"model/{args.version}",
        title=f"[Model] {args.version}",
        notes=notes,
        token=token,
    )
    print(f"  Release created: {release['html_url']}")

    print("\nUploading assets...")
    upload_url = release["upload_url"]
    for asset_path in assets:
        upload_asset(upload_url, asset_path, token)

    print(f"\nDone: {release['html_url']}")


if __name__ == "__main__":
    main()
