"""Reference heatmap decoding and NME computation.

Ported from `HRNet-Facial-Landmark-Detection/lib/core/evaluation.py`
(MIT License, Copyright (c) Microsoft).

Provides:
  - decode_preds: hard argmax + sub-pixel gradient shift + inverse affine
    transform back to 512px image space.
  - compute_nme: per-sample Normalized Mean Error using inter-ocular distance
    (landmarks 60 and 72 for WFLW 98-point).
"""

import math

import numpy as np
import torch

from ..datasets.wflw.ref_transforms import transform_preds


def _get_preds(scores):
    """Get peak locations from score maps.

    Args:
        scores: (B, K, H, W) heatmap tensor.

    Returns:
        (B, K, 2) tensor of peak locations (1-indexed).
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(
        scores.view(scores.size(0), scores.size(1), -1), 2
    )

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def decode_preds(batch_heatmaps, center, scale, heatmap_size):
    """Decode heatmaps to landmark coordinates in 512px image space.

    Applies hard argmax, sub-pixel gradient-based refinement, and inverse
    affine transform to map predictions back to the original image space.

    Args:
        batch_heatmaps: (B, K, H, W) predicted heatmaps (can be on GPU or CPU).
        center: (B, 2) tensor of face box centers.
        scale: (B,) tensor or list of scale values.
        heatmap_size: [W, H] of heatmap resolution (e.g. [64, 64]).

    Returns:
        (B, K, 2) tensor of predicted landmarks in 512px space.
    """
    coords = _get_preds(batch_heatmaps)  # (B, K, 2), 1-indexed
    coords = coords.cpu()

    # Sub-pixel refinement via gradient sign
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < heatmap_size[0]) and \
               (py > 1) and (py < heatmap_size[1]):
                diff = torch.Tensor([
                    hm[py - 1][px] - hm[py - 1][px - 2],
                    hm[py][px - 1] - hm[py - 2][px - 1],
                ])
                coords[n][p] += diff.sign() * 0.25

    coords += 0.5
    preds = coords.clone()

    # Inverse affine transform back to 512px image space
    for i in range(coords.size(0)):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], heatmap_size
        )

    if preds.dim() < 3:
        preds = preds.view(1, preds.size(0), preds.size(1))

    return preds


def compute_nme(preds, meta, iod_left=None, iod_right=None):
    """Compute per-sample Normalized Mean Error.

    NME is normalised by inter-ocular distance (IOD), computed as the
    Euclidean distance between two eye corner landmarks.

    Args:
        preds: (B, N, 2) numpy array or tensor of predicted landmarks
               in 512px space.
        meta: Dict with 'pts' key containing (B, N, 2) ground truth
              landmarks in 512px space.
        iod_left: Index of left IOD landmark within the coordinate array.
                  If None, auto-detected from landmark count.
        iod_right: Index of right IOD landmark within the coordinate array.
                   If None, auto-detected from landmark count.

    Returns:
        (B,) numpy array of per-sample NME values.
    """
    targets = meta['pts']
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
    if isinstance(targets, torch.Tensor):
        target = targets.cpu().numpy()
    else:
        target = np.array(targets)

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred = preds[i]
        pts_gt = target[i]

        if iod_left is not None and iod_right is not None:
            interocular = np.linalg.norm(pts_gt[iod_left] - pts_gt[iod_right])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        elif L == 68:
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 19:
            interocular = meta['box_size'][i]
        elif L == 29:
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        else:
            raise ValueError(
                f'Unsupported number of landmarks: {L}. '
                f'Pass iod_left and iod_right explicitly for custom landmark counts.'
            )

        if interocular <= 0:
            rmse[i] = 0.0
            continue

        rmse[i] = np.sum(
            np.linalg.norm(pts_pred - pts_gt, axis=1)
        ) / (interocular * L)

    return rmse
