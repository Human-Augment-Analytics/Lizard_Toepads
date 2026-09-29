"""
LizardDataset variant that reads the same .pt files used by the HRNet pipeline.

Each .pt file contains:
    image: (3, H, W) uint8 tensor  — raw pixel values (NOT ImageNet-normalised)
    tps:   (9, 2)   float tensor   — landmark pixel coordinates in the original
                                     image space (x, y), NOT normalised.

This lets the SHG and HRNet heatmap experiments share exactly the same data
split, making results directly comparable.

Gaussian heatmaps are generated on-the-fly at heatmap_size resolution so no
pre-processing step is needed.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A


class LizardDatasetPT(Dataset):
    """Dataset backed by .pt files (same format as HRNet heatmap pipeline).

    Args:
        pt_paths:     List of paths to .pt files.
        input_size:   Spatial size images are resized/padded to before the
                      network (default 512).
        heatmap_size: Spatial size of the output Gaussian heatmaps (default
                      128, matching the SHG output resolution).
        sigma:        Gaussian sigma in heatmap pixels (default 5.0).
    """

    def __init__(
        self,
        pt_paths,
        input_size: int = 512,
        heatmap_size: int = 128,
        sigma: float = 5.0,
    ):
        self.paths = pt_paths
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        # Augmentation pipeline — no flips (lizard pipeline removes flip
        # variance via OBB orientation), no ImageNet normalisation (SHG trains
        # from scratch with raw [0,1] pixel values).
        self.transform = A.Compose(
            [
                A.LongestMaxSize(max_size=input_size),
                A.PadIfNeeded(
                    input_size, input_size, border_mode=cv2.BORDER_CONSTANT
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=25,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.8,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=10,
                            sat_shift_limit=15,
                            val_shift_limit=10,
                        ),
                    ],
                    p=0.7,
                ),
                A.GaussNoise(p=0.3),
            ],
            keypoint_params=A.KeypointParams(
                format="xy", remove_invisible=False
            ),
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], weights_only=False)

        # image: (3, H, W) tensor → (H, W, 3) numpy for albumentations
        img = data["image"].permute(1, 2, 0).numpy()  # uint8 or float

        # coords: (9, 2) pixel coords in original image space
        coords = data["tps"].numpy().astype(np.float32)

        H, W = img.shape[:2]
        coords[:, 0] = np.clip(coords[:, 0], 0, W - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, H - 1)

        # Retry augmentation up to 10 times to avoid losing keypoints off-edge
        keypoints = coords.tolist()
        for _ in range(10):
            aug = self.transform(image=img, keypoints=keypoints)
            kp = np.array(aug["keypoints"], dtype=np.float32)
            if kp.shape[0] == 9:
                break
        else:
            aug = self.transform(image=img, keypoints=keypoints)
            kp = np.array(keypoints, dtype=np.float32)

        img_aug = aug["image"]  # (input_size, input_size, 3)

        kp[:, 0] = np.clip(kp[:, 0], 0, self.input_size - 1)
        kp[:, 1] = np.clip(kp[:, 1], 0, self.input_size - 1)

        # Normalise coordinates to [0, 1] for heatmap generation
        coords_norm = kp / self.input_size  # (9, 2)

        # Build Gaussian heatmaps at heatmap_size resolution
        heatmaps = _make_gaussian_heatmaps(
            coords_norm, self.heatmap_size, self.sigma
        )  # (9, heatmap_size, heatmap_size)

        # Image: (H, W, C) → (H, W, C) float [0, 1]
        # SHG forward() expects (B, H, W, C) and internally permutes to (B, C, H, W)
        img_tensor = torch.from_numpy(img_aug).float() / 255.0  # (H, W, 3)

        return img_tensor, heatmaps


def _make_gaussian_heatmaps(
    coords_norm: np.ndarray,
    heatmap_size: int,
    sigma: float,
) -> torch.Tensor:
    """Generate (K, heatmap_size, heatmap_size) Gaussian heatmaps.

    Args:
        coords_norm: (K, 2) float32 array, values in [0, 1].
        heatmap_size: Output spatial size (square).
        sigma: Gaussian sigma in heatmap pixels.

    Returns:
        Tensor of shape (K, heatmap_size, heatmap_size), values in [0, 1].
    """
    K = coords_norm.shape[0]
    H = W = heatmap_size

    # Pixel-space landmark positions on the heatmap grid
    px = coords_norm[:, 0] * (W - 1)  # (K,)
    py = coords_norm[:, 1] * (H - 1)  # (K,)

    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)  # both (H, W)

    # (K, H, W)
    dx = grid_x[None] - px[:, None, None]
    dy = grid_y[None] - py[:, None, None]
    heatmaps = np.exp(-(dx**2 + dy**2) / (2 * sigma**2)).astype(np.float32)

    return torch.from_numpy(heatmaps)
