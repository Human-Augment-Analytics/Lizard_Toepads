"""Path resolution and environment variable overrides for config.

Environment variables with the prefix LANDMARKING_ override corresponding
path fields in the configuration.
"""

import os
from pathlib import Path

ENV_PREFIX = "LANDMARKING_"

# Maps dataset names to their conventional directory names
DATASET_DIR_MAPPING = {
    "lizard": "Lizard_data",
    "wflw": "WFLW_data",
    "cephalometric": "Cephalometric_data",
}


def resolve_path(field_name: str, default: str) -> str:
    """Check for LANDMARKING_{FIELD_NAME} env var, fall back to default.

    Args:
        field_name: The configuration field name (e.g., "DATA_ROOT").
        default: The default value to use if env var is not set.

    Returns:
        The resolved path string.
    """
    env_key = f"{ENV_PREFIX}{field_name.upper()}"
    return os.environ.get(env_key, default)


def resolve_dataset_dir(data_root: str, dataset_name: str) -> str:
    """Resolve dataset directory from data_root and dataset name.

    Maps known dataset names to their conventional subdirectory names.
    Unknown dataset names are used as-is. Uses POSIX-style path joining
    for cross-platform consistency (paths target Linux HPC).

    Args:
        data_root: The root data directory.
        dataset_name: The dataset name (e.g., "lizard", "wflw").

    Returns:
        Full path to the dataset directory (POSIX separators).
    """
    subdir = DATASET_DIR_MAPPING.get(dataset_name, dataset_name)
    # Use forward slashes for cross-platform consistency
    root = data_root.rstrip("/")
    return f"{root}/{subdir}"
