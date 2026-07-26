"""TPS file I/O utilities for Lizard dataset.

Reads .TPS morphometric landmark files and extracts coordinates.
Also provides ruler distance extraction for pixel-to-mm conversion.
"""

import numpy as np


def get_tps_coords(img_id: str, img: np.ndarray, tps_data_dir: str) -> dict:
    """Read TPS landmark coordinates for both finger and toe classes.

    TPS files contain ruler points (first 2) followed by anatomical landmarks.
    The ruler points are skipped; only anatomical landmarks are returned.
    Y coordinates are flipped (TPS uses bottom-left origin).

    Args:
        img_id: Image identifier (numeric string).
        img: Image array (used for height to flip Y axis).
        tps_data_dir: Directory containing TPS files.

    Returns:
        Dict mapping class name ("finger", "toe") to list of (x, y) tuples.
    """
    cls = ["finger", "toe"]
    ret = {}
    h, w = img.shape[:2]
    for c in cls:
        fp = f"{tps_data_dir}/{img_id}_{c}.TPS"
        coordinates = []
        skip = 2  # Skip first 2 coordinate pairs (ruler points)
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line or "=" in line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    if skip > 0:
                        skip -= 1
                        continue
                    try:
                        x, y = map(float, parts)
                        coordinates.append((x, h - 1 - y))
                    except ValueError:
                        continue
        ret[c] = coordinates
    return ret


def get_ruler_distance(img_id: str, img: np.ndarray, tps_data_dir: str) -> dict:
    """Return the pixel distance between the two ruler landmarks for each class.

    These are the first two coordinate pairs in each TPS file (the ones
    skipped by get_tps_coords). The distance can be used to convert pixel
    error to mm given a known physical ruler length.

    Args:
        img_id: Image identifier (numeric string).
        img: Image array (used for height to flip Y axis).
        tps_data_dir: Directory containing TPS files.

    Returns:
        Dict mapping class name to ruler pixel distance (or None if unavailable).
    """
    cls = ["finger", "toe"]
    ret = {}
    h, w = img.shape[:2]
    for c in cls:
        fp = f"{tps_data_dir}/{img_id}_{c}.TPS"
        ruler_pts = []
        try:
            with open(fp, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or "=" in line:
                        continue
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            x, y = map(float, parts)
                            ruler_pts.append(np.array([x, h - 1 - y]))
                        except ValueError:
                            continue
                        if len(ruler_pts) == 2:
                            break
        except FileNotFoundError:
            ret[c] = None
            continue
        if len(ruler_pts) == 2:
            ret[c] = float(np.linalg.norm(ruler_pts[0] - ruler_pts[1]))
        else:
            ret[c] = None
    return ret
