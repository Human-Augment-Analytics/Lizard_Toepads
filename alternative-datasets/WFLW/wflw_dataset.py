"""
WFLW dataset class — compatible with the existing LizardDataset .pt format
but supports any number of landmarks (not hardcoded to 9).

Drop-in replacement for LizardDataset when training on WFLW.

Coordinate convention:
  preprocess.py saves "tps" as (N, 2) float32 in [0, 1], normalized to the
  512x512 letterboxed image. This class reads them directly — no re-scaling.

Augmentation:
  Geometric augmentation requires landmark-aware handling:
    - HorizontalFlip: x coords mirrored AND landmarks reordered via WFLW_FLIP_PAIRS
    - Affine (scale/translate only, no rotation): albumentations handles keypoints
    - No rotation: the mean-shape GCN init assumes upright faces; rotation would
      require rotating the mean shape too, deferred to a future experiment.
  Image-only augmentation (brightness/contrast) needs no special handling.
"""
import sys
from pathlib import Path
import random

import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
import numpy as np

# Import WFLW_FLIP_PAIRS from graph_topology.py in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from graph_topology import WFLW_FLIP_PAIRS


def _build_flip_permutation(flip_pairs: list, num_landmarks: int) -> np.ndarray:
    """Convert flip-pair list to a permutation index array.

    Returns perm such that flipped_coords = coords[perm] after mirroring x.
    """
    perm = np.arange(num_landmarks, dtype=np.int64)
    for i, j in flip_pairs:
        perm[i] = j
        perm[j] = i
    return perm


# Pre-build the 98-point permutation at import time
_FLIP_PERM_98 = _build_flip_permutation(WFLW_FLIP_PAIRS, 98)


class WFLWDataset(Dataset):
    def __init__(self, pt_paths, input_size=512, num_landmarks=98,
                 augment=True, flip_prob=0.5):
        """
        Args:
            pt_paths:      List of .pt file paths.
            input_size:    Expected image size (square). .pt files are already
                           letterbox-cropped to this size by preprocess.py.
            num_landmarks: Number of landmarks (default 98 for WFLW).
            augment:       Whether to apply augmentation (disable for val/test).
            flip_prob:     Probability of horizontal flip when augment=True.
        """
        self.paths = pt_paths
        self.input_size = input_size
        self.num_landmarks = num_landmarks
        self.augment = augment
        self.flip_prob = flip_prob

        # Use the pre-built 98-point permutation or build one for other counts
        if num_landmarks == 98:
            self.flip_perm = _FLIP_PERM_98
        else:
            # Fallback: no flip reordering for non-98-point schemes
            self.flip_perm = np.arange(num_landmarks, dtype=np.int64)

        # Image-only augmentation (no landmark impact)
        self.color_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        ])

        # Resize transform — applied when input_size != 512 (the native .pt crop size)
        self.resize_transform = None
        if input_size != 512:
            import cv2 as _cv2
            self.resize_transform = A.Compose([
                A.LongestMaxSize(max_size=input_size),
                A.PadIfNeeded(input_size, input_size, border_mode=_cv2.BORDER_CONSTANT),
            ])

        self.normalize = A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx])
        img = data["image"].permute(1, 2, 0).numpy()    # HWC uint8, already 512x512
        coords = data["tps"].numpy().copy()              # (N, 2) float32, already [0,1]
        orig_size = data.get("orig_size", torch.tensor([img.shape[0], img.shape[1]]))

        was_flipped = False
        if self.augment:
            img, coords, was_flipped = self._augment(img, coords)

        # Resize to input_size if different from native 512px crop size
        if self.resize_transform is not None:
            resized = self.resize_transform(image=img)
            img = resized["image"]
            # Coords are already in [0,1] relative to 512px; they remain valid
            # after letterbox resize since the relative positions are preserved.

        img_norm = self.normalize(image=img)["image"]        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        coords_tensor = torch.from_numpy(coords).float()
        flipped_tensor = torch.tensor(was_flipped, dtype=torch.bool)

        return img_tensor, coords_tensor, orig_size, flipped_tensor

    def _augment(self, img: np.ndarray, coords: np.ndarray):
        """Apply augmentation. Returns (img, coords, was_flipped)."""
        was_flipped = False

        # ── Horizontal flip ───────────────────────────────────────────────
        if random.random() < self.flip_prob:
            img = img[:, ::-1, :].copy()
            coords[:, 0] = 1.0 - coords[:, 0]
            coords = coords[self.flip_perm]
            was_flipped = True

        # ── Affine (scale + translate, no rotation) ───────────────────────
        # Implemented manually so we can reject transforms that push any
        # landmark out of [0,1] bounds rather than clipping — clipping
        # multiple landmarks to the same border coordinate causes centroid
        # collapse during training.
        if random.random() < 0.7:
            for _ in range(10):
                scale = random.uniform(0.90, 1.10)
                tx = random.uniform(-0.05, 0.05)  # fraction of image width
                ty = random.uniform(-0.05, 0.05)

                # Transform coords: scale around centre, then translate
                new_coords = (coords - 0.5) * scale + 0.5
                new_coords[:, 0] += tx
                new_coords[:, 1] += ty

                # Only accept if all landmarks stay within bounds
                if new_coords.min() >= 0.0 and new_coords.max() <= 1.0:
                    # Apply same transform to image via OpenCV
                    cx, cy = self.input_size / 2, self.input_size / 2
                    M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
                    M[0, 2] += tx * self.input_size
                    M[1, 2] += ty * self.input_size
                    img = cv2.warpAffine(
                        img, M, (self.input_size, self.input_size),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    coords = new_coords
                    break
            # If no valid transform found after 10 tries, skip affine

        # ── Color jitter ──────────────────────────────────────────────────
        img = self.color_transform(image=img)["image"]

        return img, coords, was_flipped
