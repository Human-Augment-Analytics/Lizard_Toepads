"""
Shared split utilities for fraction sampling and split file writing.
Used by both the Lizard and WFLW split generators.
"""
import json
import math
import random
from pathlib import Path


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
        raise ValueError(
            f"fraction must be in (0, 1], got {fraction}"
        )
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
        "train": [str(p) for p in train],
        "val": [str(p) for p in val],
        "test": [str(p) for p in test],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(split, f, indent=2)
