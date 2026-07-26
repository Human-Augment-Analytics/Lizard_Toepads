"""Deterministic split generation for landmark detection datasets.

Provides utilities for creating reproducible train/val/test splits
with configurable fractions and optional predefined test sets.
"""

import json
import math
import random
from pathlib import Path
from typing import Optional


def generate_split(
    data_dir: str,
    fractions: dict,
    seed: int = 42,
    output_path: Optional[str] = None,
    predefined_test: Optional[list] = None,
    glob_pattern: str = "*.pt",
) -> dict:
    """Generate deterministic train/val/test split.

    Uses random.Random(seed) for all randomness, ensuring reproducibility.
    Writes JSON if output_path is provided. JSON uses sorted keys for
    byte-identical reproducibility.

    Args:
        data_dir: Directory containing data files to split.
        fractions: Dict with keys 'train', 'val', 'test' mapping to floats.
                   Values should sum to <= 1.0.
        seed: Random seed for reproducibility.
        output_path: If provided, write the split JSON to this path.
        predefined_test: If provided, these paths are used as the test set
                         and excluded from train/val sampling (for WFLW).
        glob_pattern: Glob pattern to find files in data_dir.

    Returns:
        Dict with keys 'train', 'val', 'test' each mapping to sorted lists
        of path strings.
    """
    data_path = Path(data_dir)
    all_paths = sorted([str(p) for p in data_path.glob(glob_pattern)])

    rng = random.Random(seed)

    if predefined_test is not None:
        test_set = sorted(predefined_test)
        test_set_s = set(test_set)
        remaining = [p for p in all_paths if p not in test_set_s]
    else:
        test_frac = fractions.get("test", 0.0)
        n_test = math.floor(len(all_paths) * test_frac)
        test_set = sorted(rng.sample(all_paths, n_test))
        test_set_s = set(test_set)
        remaining = [p for p in all_paths if p not in test_set_s]

    train_frac = fractions.get("train", 0.8)
    val_frac = fractions.get("val", 0.1)

    # Compute sizes from the original total
    n_total = len(all_paths)
    n_train = math.floor(n_total * train_frac)
    n_val = math.floor(n_total * val_frac)

    # Shuffle remaining and assign
    rng.shuffle(remaining)
    train_set = sorted(remaining[:n_train])
    val_set = sorted(remaining[n_train:n_train + n_val])

    split = {
        "test": test_set,
        "train": train_set,
        "val": val_set,
    }

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(split, f, indent=2, sort_keys=True)

    return split


def sample_fraction(paths: list, fraction: float, seed: int) -> list:
    """Return floor(fraction * len(paths)) items sampled without replacement.

    Deterministic for a fixed (paths, fraction, seed) triple.

    Args:
        paths: List of path strings to sample from.
        fraction: Fraction of paths to return. Must be in (0, 1].
        seed: Random seed for reproducibility.

    Returns:
        A list of sampled paths of length floor(fraction * len(paths)).

    Raises:
        ValueError: If fraction is not in (0, 1].
    """
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    n = math.floor(fraction * len(paths))
    rng = random.Random(seed)
    return rng.sample(paths, n)


def write_split(train: list, val: list, test: list, output_path: str) -> None:
    """Write a split JSON file with train, val, and test keys.

    Args:
        train: List of training file paths.
        val: List of validation file paths.
        test: List of test file paths.
        output_path: Path to write the JSON file.
    """
    split = {
        "train": sorted([str(p) for p in train]),
        "val": sorted([str(p) for p in val]),
        "test": sorted([str(p) for p in test]),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(split, f, indent=2, sort_keys=True)
