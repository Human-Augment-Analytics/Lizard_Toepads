"""Test that mean shape subsampling produces correct spatial layout.

Verifies that after subsampling the 98-point mean shape to 25 points,
the jaw landmarks are at the face contour (not clustered at center),
and eye landmarks are at the correct eye positions.
"""

import torch
import numpy as np
from pathlib import Path

from landmarking.common.sparsity import make_subset_indices


def test_mean_shape_subsample_spatial_layout():
    """Verify subsampled mean shape has correct face layout."""
    # Load the actual mean shape
    mean_shape_path = Path("./runs/wflw/mean_shapes/mean_shape_wflw.pt")
    if not mean_shape_path.exists():
        # Try alternative path
        mean_shape_path = Path("/home/hice1/axu39/scratch/WFLW_data/mean_shapes/mean_shape_wflw.pt")
    if not mean_shape_path.exists():
        print("SKIP: mean shape file not found")
        return

    mean_shape = torch.load(str(mean_shape_path), map_location="cpu")
    print(f"Full mean shape: {mean_shape.shape}")
    print(f"Full mean shape range: x=[{mean_shape[:, 0].min():.3f}, {mean_shape[:, 0].max():.3f}], y=[{mean_shape[:, 1].min():.3f}, {mean_shape[:, 1].max():.3f}]")

    # Print jaw landmarks (0-32) in the full shape
    print(f"\nFull shape jaw landmarks (0-32):")
    for i in [0, 8, 16, 24, 32]:
        print(f"  LM {i}: ({mean_shape[i, 0]:.3f}, {mean_shape[i, 1]:.3f})")

    # Print eye landmarks
    print(f"\nFull shape eye landmarks:")
    print(f"  LM 60 (right eye inner): ({mean_shape[60, 0]:.3f}, {mean_shape[60, 1]:.3f})")
    print(f"  LM 72 (left eye outer):  ({mean_shape[72, 0]:.3f}, {mean_shape[72, 1]:.3f})")

    # Now subsample
    indices = make_subset_indices(total=98, step=4)
    print(f"\nSubset indices ({len(indices)} landmarks): {indices}")

    subsampled = mean_shape[indices]
    print(f"\nSubsampled mean shape: {subsampled.shape}")
    print(f"Subsampled range: x=[{subsampled[:, 0].min():.3f}, {subsampled[:, 0].max():.3f}], y=[{subsampled[:, 1].min():.3f}, {subsampled[:, 1].max():.3f}]")

    # Check jaw landmarks in subset (positions 0-8 for LM 0,4,8,12,16,20,24,28,32)
    print(f"\nSubsampled jaw landmarks (subset positions 0-8):")
    jaw_x_min = float('inf')
    jaw_x_max = float('-inf')
    for i in range(9):  # First 9 in subset are jaw
        x, y = subsampled[i, 0].item(), subsampled[i, 1].item()
        orig_lm = indices[i]
        print(f"  Subset pos {i} (LM {orig_lm}): ({x:.3f}, {y:.3f})")
        jaw_x_min = min(jaw_x_min, x)
        jaw_x_max = max(jaw_x_max, x)

    # Jaw should span a wide x range (face contour)
    jaw_x_spread = jaw_x_max - jaw_x_min
    print(f"\nJaw x-spread: {jaw_x_spread:.3f} (should be > 0.5 for proper face contour)")

    # Check eye landmarks in subset
    pos_60 = indices.index(60)
    pos_72 = indices.index(72)
    print(f"\nEye landmarks in subset:")
    print(f"  Subset pos {pos_60} (LM 60): ({subsampled[pos_60, 0]:.3f}, {subsampled[pos_60, 1]:.3f})")
    print(f"  Subset pos {pos_72} (LM 72): ({subsampled[pos_72, 0]:.3f}, {subsampled[pos_72, 1]:.3f})")

    # IOD distance
    iod = torch.norm(subsampled[pos_60] - subsampled[pos_72]).item()
    print(f"  IOD (in [0,1] space): {iod:.3f} (should be ~0.15-0.3)")

    # ASSERTIONS
    assert jaw_x_spread > 0.4, f"Jaw x-spread too small: {jaw_x_spread:.3f} — mean shape is probably wrong"
    assert iod > 0.05, f"IOD too small: {iod:.3f} — eye landmarks are too close"
    assert subsampled[:, 0].min() < 0.3, "No landmarks on the left side of face"
    assert subsampled[:, 0].max() > 0.7, "No landmarks on the right side of face"

    print("\n✓ All spatial layout checks PASSED")


if __name__ == "__main__":
    test_mean_shape_subsample_spatial_layout()
