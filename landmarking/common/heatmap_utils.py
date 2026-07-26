"""Heatmap generation and coordinate extraction utilities.

Provides Gaussian heatmap generation, soft/hard argmax decoding,
and visualization overlay generation.
"""

import numpy as np
import torch
import torch.nn.functional as F


def generate_gaussian_heatmap(
    coords: np.ndarray,
    heatmap_size: int,
    sigma: float = 4.0,
) -> np.ndarray:
    """Generate Gaussian heatmaps from landmark coordinates.

    Args:
        coords: (num_landmarks, 2) array of (x, y) coordinates in pixel space
                relative to the heatmap resolution, OR normalized [0, 1].
        heatmap_size: Square heatmap side length.
        sigma: Gaussian sigma in heatmap pixels.

    Returns:
        (num_landmarks, heatmap_size, heatmap_size) float32 array with peaks at 1.0.
    """
    num_landmarks = coords.shape[0]
    H = W = heatmap_size
    heatmaps = np.zeros((num_landmarks, H, W), dtype=np.float32)

    yy, xx = np.mgrid[0:H, 0:W]

    for i in range(num_landmarks):
        x, y = float(coords[i, 0]), float(coords[i, 1])
        # If coords are normalized [0,1], scale to heatmap pixels
        if 0 <= x <= 1 and 0 <= y <= 1:
            x = x * (W - 1)
            y = y * (H - 1)

        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        heatmaps[i] = g

    return heatmaps


def soft_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """Differentiable spatial expectation (soft-argmax).

    Args:
        heatmaps: (B, K, H, W) raw logits.

    Returns:
        (B, K, 2) expected coordinates in [0, 1].
    """
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    weights = F.softmax(flat, dim=-1).view(B, K, H, W)

    device = heatmaps.device
    xs = torch.linspace(0, 1, W, device=device)
    ys = torch.linspace(0, 1, H, device=device)

    x_coords = (weights.sum(dim=2) * xs.view(1, 1, W)).sum(dim=-1)
    y_coords = (weights.sum(dim=3) * ys.view(1, 1, H)).sum(dim=-1)

    return torch.stack([x_coords, y_coords], dim=-1)


def hard_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """Hard argmax with sub-pixel refinement.

    Matches paper's decode_preds: find peak pixel, then shift ±0.25px
    based on local gradient direction around the peak.

    Args:
        heatmaps: (B, K, H, W) raw logits.

    Returns:
        (B, K, 2) peak locations in [0, 1].
    """
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    idx = flat.argmax(dim=-1)
    py = idx // W
    px = idx % W

    # Sub-pixel refinement
    px_c = px.clamp(1, W - 2)
    py_c = py.clamp(1, H - 2)

    b_idx = torch.arange(B, device=heatmaps.device).unsqueeze(1).expand(B, K)
    k_idx = torch.arange(K, device=heatmaps.device).unsqueeze(0).expand(B, K)

    def g(r, c):
        return heatmaps[b_idx, k_idx, r.clamp(0, H - 1), c.clamp(0, W - 1)]

    dx = (g(py_c - 1, px_c) - g(py_c - 1, px_c - 2)).sign() * 0.25
    dy = (g(py_c, px_c - 1) - g(py_c - 2, px_c - 1)).sign() * 0.25

    x_refined = (px.float() + dx + 0.5) / W
    y_refined = (py.float() + dy + 0.5) / H

    return torch.stack([x_refined, y_refined], dim=-1)


def generate_overlay(
    image: np.ndarray,
    pred_coords: np.ndarray,
    gt_coords: np.ndarray = None,
    radius: int = 3,
) -> np.ndarray:
    """Generate visualization overlay with landmarks on image.

    Args:
        image: (H, W, 3) uint8 BGR/RGB image.
        pred_coords: (N, 2) predicted landmark coordinates in pixel space.
        gt_coords: (N, 2) ground truth coordinates (optional).
        radius: Circle radius for landmark markers.

    Returns:
        (H, W, 3) uint8 image with landmarks drawn.
    """
    import cv2

    overlay = image.copy()
    H, W = overlay.shape[:2]

    # Draw predictions in red
    for i in range(pred_coords.shape[0]):
        x, y = int(pred_coords[i, 0]), int(pred_coords[i, 1])
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(overlay, (x, y), radius, (0, 0, 255), -1)

    # Draw ground truth in green
    if gt_coords is not None:
        for i in range(gt_coords.shape[0]):
            x, y = int(gt_coords[i, 0]), int(gt_coords[i, 1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(overlay, (x, y), radius, (0, 255, 0), -1)

    return overlay
