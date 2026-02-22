#!/usr/bin/env python3
"""
Download YOLO pre-trained models (YOLOv11 detect + YOLO26 OBB) to models/base_models/
"""

import os
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

# Model configurations: (filename, size_mb)
MODELS: Dict[str, Tuple[str, float]] = {
    # Detection models
    "yolov11n.pt": ("yolo11n.pt", 5.4),
    "yolov11s.pt": ("yolo11s.pt", 18.5),
    "yolov11m.pt": ("yolo11m.pt", 40.2),
    "yolov11l.pt": ("yolo11l.pt", 49.7),
    "yolov11x.pt": ("yolo11x.pt", 109.3),
    # YOLO26 OBB (Oriented Bounding Box) models
    "yolo26n-obb.pt": ("yolo26n-obb.pt", 5.0),
    "yolo26s-obb.pt": ("yolo26s-obb.pt", 18.0),
    "yolo26m-obb.pt": ("yolo26m-obb.pt", 38.0),
    "yolo26l-obb.pt": ("yolo26l-obb.pt", 48.0),
    "yolo26x-obb.pt": ("yolo26x-obb.pt", 110.0),
}

BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"


def download_file(url: str, dest_path: Path, expected_size_mb: float) -> bool:
    """Download a file with progress reporting"""
    try:
        print(f"Downloading {dest_path.name} (~{expected_size_mb}MB)...", end="")

        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            if block_num % 100 == 0:  # Update every 100 blocks
                print(f"\rDownloading {dest_path.name} (~{expected_size_mb}MB)... {percent:.1f}%", end="")

        urllib.request.urlretrieve(url, dest_path, reporthook=download_progress)

        # Verify file size
        actual_size_mb = dest_path.stat().st_size / (1024 * 1024)
        if actual_size_mb < 1:  # File too small, likely an error page
            print(f" [FAIL] Failed (file too small: {actual_size_mb:.1f}MB)")
            dest_path.unlink()
            return False

        print(f" [OK] Done ({actual_size_mb:.1f}MB)")
        return True

    except Exception as e:
        print(f" [FAIL] Failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def main():
    # Create models directory
    models_dir = Path("models/base_models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("YOLO Model Downloader (YOLOv11 Detect + YOLO26 OBB)")
    print("=" * 50)
    print(f"Download directory: {models_dir.absolute()}")
    print()

    # Check existing models
    existing = []
    for local_name, (remote_name, size_mb) in MODELS.items():
        local_path = models_dir / local_name
        if local_path.exists():
            actual_size_mb = local_path.stat().st_size / (1024 * 1024)
            existing.append(f"  - {local_name} ({actual_size_mb:.1f}MB)")

    if existing:
        print("Existing models found:")
        for model in existing:
            print(model)
        print()

    # Download missing models
    downloaded = []
    skipped = []
    failed = []

    for local_name, (remote_name, size_mb) in MODELS.items():
        local_path = models_dir / local_name

        if local_path.exists():
            actual_size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"Skipping {local_name} (already exists: {actual_size_mb:.1f}MB)")
            skipped.append(local_name)
            continue

        url = BASE_URL + remote_name
        if download_file(url, local_path, size_mb):
            downloaded.append(local_name)
        else:
            failed.append(local_name)

    # Summary
    print()
    print("=" * 50)
    print("Summary:")
    print(f"  Downloaded: {len(downloaded)} models")
    print(f"  Skipped (existing): {len(skipped)} models")
    print(f"  Failed: {len(failed)} models")

    if failed:
        print("\nFailed downloads:")
        for model in failed:
            print(f"  - {model}")
        print("\nYou can try downloading these manually using curl commands from the README.")

    print("\nAll available models:")
    for local_name in MODELS.keys():
        local_path = models_dir / local_name
        if local_path.exists():
            actual_size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {local_name} ({actual_size_mb:.1f}MB)")
        else:
            print(f"  [MISSING] {local_name} (not downloaded)")


if __name__ == "__main__":
    main()