"""Reader for consolidated (multi-specimen) TPS morphometric files.

Lives in ``common`` rather than ``datasets.lizard`` so that it, and the shape
analysis built on it, can be imported without pulling in torch: the
``landmarking.datasets`` package imports torch at module scope via
``datasets.base``. ``datasets.lizard.tps_utils`` re-exports this for
discoverability alongside the per-image TPS helpers.

Pure numpy, mirroring ``metrics_lizard.py`` and ``shape_space.py``.
"""

import numpy as np

__all__ = ["read_consolidated_tps"]


def read_consolidated_tps(path: str, expected_landmarks: int = 9) -> list:
    """Read a multi-specimen (consolidated) TPS file.

    Unlike ``datasets.lizard.tps_utils.get_tps_coords``, which expects one
    ``{img_id}_{class}.TPS`` per specimen and needs the image to flip the Y
    axis, this reads the consolidated format where many specimens are
    concatenated::

        LM=9
        7636.00000 2581.00000
        ...
        IMAGE=1003.jpg

    Blocks declaring more landmarks than ``expected_landmarks`` are assumed to
    carry leading ruler points (the same convention ``get_tps_coords`` encodes
    as ``skip = 2``). Those extras are stripped off the front and their
    separation is returned as ``ruler_px``.

    Y is NOT flipped, because no image height is available here. That is safe
    for shape analysis: flipping Y applies the *same* reflection to every
    specimen, and Procrustes alignment, PCA variance structure, and
    reconstruction error are all invariant to a global reflection of the whole
    sample. Do not use this function where absolute image-frame orientation
    matters.

    Args:
        path: Path to the consolidated .TPS/.tps file.
        expected_landmarks: Number of anatomical landmarks per specimen.

    Returns:
        List of dicts, one per specimen, each with:
            ``landmarks``: (expected_landmarks, 2) float64 array.
            ``ruler_px``: float separation of the stripped ruler points, or
                None when the block had no extras.
            ``image``: value of the trailing ``IMAGE=`` line, or None.
        Specimens whose landmark count does not match after stripping are
        skipped.

    Raises:
        ValueError: If ``expected_landmarks`` is not positive.
    """
    if expected_landmarks <= 0:
        raise ValueError(
            f"expected_landmarks must be positive, got {expected_landmarks}"
        )

    specimens = []
    state = {"declared": None, "points": []}

    def _flush(image_name):
        declared = state["declared"]
        points = state["points"]
        if declared is None or len(points) != declared:
            return
        arr = np.asarray(points, dtype=np.float64)
        extra = declared - expected_landmarks
        ruler_px = None
        if extra > 0:
            ruler = arr[:extra]
            arr = arr[extra:]
            if extra >= 2:
                ruler_px = float(np.linalg.norm(ruler[0] - ruler[1]))
        if arr.shape[0] != expected_landmarks:
            return
        specimens.append(
            {"landmarks": arr, "ruler_px": ruler_px, "image": image_name}
        )

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("LM="):
                # A new block starts; any previous one had no IMAGE= line.
                _flush(None)
                try:
                    state["declared"] = int(line.split("=", 1)[1])
                except ValueError:
                    state["declared"] = None
                state["points"] = []
            elif upper.startswith("IMAGE="):
                _flush(line.split("=", 1)[1])
                state["declared"] = None
                state["points"] = []
            elif "=" in line:
                # ID=, SCALE=, COMMENT= and friends: not coordinates.
                continue
            else:
                parts = line.split()
                if len(parts) == 2 and state["declared"] is not None:
                    try:
                        state["points"].append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue

    _flush(None)
    return specimens
