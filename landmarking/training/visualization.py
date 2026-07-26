"""Training visualization utilities.

Generates overlay images showing predicted and ground truth landmarks
on input images during training for monitoring convergence.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

# ImageNet normalization constants for denormalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def save_overlay(
    img_tensor: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    save_path: str,
    input_size: int = 512,
) -> None:
    """Draw predicted (red) and GT (green) landmarks on the image and save.

    The image tensor is assumed to be ImageNet-normalized; this function
    reverses the normalization before drawing.

    Args:
        img_tensor: (3, H, W) float tensor, ImageNet-normalized.
        pred_coords: (N, 2) predicted coordinates in [0, 1].
        gt_coords: (N, 2) ground truth coordinates in [0, 1].
        save_path: Output file path for the overlay image.
        input_size: Image canvas size for coordinate scaling.
    """
    # Denormalize image
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # HWC float
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pred = pred_coords.cpu().numpy() * input_size
    gt = gt_coords.cpu().numpy() * input_size

    # Draw GT in green
    for x, y in gt:
        cv2.circle(img_bgr, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Draw predictions in red
    for x, y in pred:
        cv2.circle(img_bgr, (int(x), int(y)), 2, (0, 0, 255), -1)

    # Ensure output directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img_bgr)


def save_training_overlays(
    imgs: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    output_dir: str,
    epoch: int,
    n_samples: int = 3,
    input_size: int = 512,
) -> None:
    """Save overlay visualizations for a batch of samples.

    Args:
        imgs: (B, 3, H, W) batch of ImageNet-normalized images.
        pred_coords: (B, N, 2) predicted coordinates in [0, 1].
        gt_coords: (B, N, 2) ground truth coordinates in [0, 1].
        output_dir: Base directory for visualizations.
        epoch: Current epoch number.
        n_samples: Number of samples to save.
        input_size: Image canvas size.
    """
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    n = min(n_samples, imgs.shape[0])
    for i in range(n):
        save_path = vis_dir / f"overlay_epoch{epoch:04d}_sample{i}.jpg"
        save_overlay(
            imgs[i], pred_coords[i], gt_coords[i], str(save_path), input_size
        )
