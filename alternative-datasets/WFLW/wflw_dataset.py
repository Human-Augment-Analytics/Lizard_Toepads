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
    - Affine (scale/translate, no rotation by default)
    - Rotation (optional via rot_factor > 0): both image and coords rotated
      by the same angle. The training loop must also rotate the mean shape
      initialization by the same angle — the rotation angle is returned as
      the 5th element of the tuple when rot_factor > 0.
  Image-only augmentation (brightness/contrast) needs no special handling.
"""
import sys
import math
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
                 augment=True, flip_prob=0.5, rot_factor=0, rot_prob=0.6):
        """
        Args:
            pt_paths:      List of .pt file paths.
            input_size:    Expected image size (square).
            num_landmarks: Number of landmarks (default 98 for WFLW).
            augment:       Whether to apply augmentation (disable for val/test).
            flip_prob:     Probability of horizontal flip when augment=True.
            rot_factor:    Max rotation angle in degrees (0 = disabled).
                           When > 0, the dataset returns a 5-tuple:
                           (img, coords, orig_size, flipped, rot_angle_deg)
                           and the training loop must rotate the mean shape
                           by the same angle before passing to the GCN.
            rot_prob:      Probability of applying rotation when rot_factor > 0.
        """
        self.paths = pt_paths
        self.input_size = input_size
        self.num_landmarks = num_landmarks
        self.augment = augment
        self.flip_prob = flip_prob
        self.rot_factor = rot_factor
        self.rot_prob = rot_prob

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
        rot_angle = 0.0
        if self.augment:
            img, coords, was_flipped, rot_angle = self._augment(img, coords)

        # Resize to input_size if different from native 512px crop size
        if self.resize_transform is not None:
            resized = self.resize_transform(image=img)
            img = resized["image"]

        img_norm = self.normalize(image=img)["image"]
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        coords_tensor = torch.from_numpy(coords).float()
        flipped_tensor = torch.tensor(was_flipped, dtype=torch.bool)

        if self.rot_factor > 0:
            return img_tensor, coords_tensor, orig_size, flipped_tensor, torch.tensor(rot_angle, dtype=torch.float32)
        return img_tensor, coords_tensor, orig_size, flipped_tensor

    def _augment(self, img: np.ndarray, coords: np.ndarray):
        """Apply augmentation. Returns (img, coords, was_flipped, rot_angle_deg)."""
        was_flipped = False
        rot_angle = 0.0

        # ── Horizontal flip ───────────────────────────────────────────────
        if random.random() < self.flip_prob:
            img = img[:, ::-1, :].copy()
            coords[:, 0] = 1.0 - coords[:, 0]
            coords = coords[self.flip_perm]
            was_flipped = True

        # ── Affine (scale + translate) ────────────────────────────────────
        # Scale range matches the reference pipeline (SCALE_FACTOR: 0.25 → ±25%).
        if random.random() < 0.7:
            for _ in range(10):
                scale = random.uniform(0.75, 1.25)
                tx = random.uniform(-0.05, 0.05)
                ty = random.uniform(-0.05, 0.05)
                new_coords = (coords - 0.5) * scale + 0.5
                new_coords[:, 0] += tx
                new_coords[:, 1] += ty
                if new_coords.min() >= 0.0 and new_coords.max() <= 1.0:
                    cx, cy = self.input_size / 2, self.input_size / 2
                    M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
                    M[0, 2] += tx * self.input_size
                    M[1, 2] += ty * self.input_size
                    img = cv2.warpAffine(
                        img, M, (self.input_size, self.input_size),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                    )
                    coords = new_coords
                    break

        # ── In-plane rotation (optional) ──────────────────────────────────
        if self.rot_factor > 0 and random.random() < self.rot_prob:
            for _ in range(10):
                angle = random.uniform(-self.rot_factor, self.rot_factor)
                theta = math.radians(angle)
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                # Rotate coords around image centre (0.5, 0.5)
                x = coords[:, 0] - 0.5
                y = coords[:, 1] - 0.5
                x_rot = cos_t * x - sin_t * y + 0.5
                y_rot = sin_t * x + cos_t * y + 0.5
                new_coords = np.stack([x_rot, y_rot], axis=-1).astype(np.float32)
                if new_coords.min() >= 0.0 and new_coords.max() <= 1.0:
                    cx, cy = self.input_size / 2, self.input_size / 2
                    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
                    img = cv2.warpAffine(
                        img, M, (self.input_size, self.input_size),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                    )
                    coords = new_coords
                    rot_angle = angle
                    break

        # ── Color jitter ──────────────────────────────────────────────────
        img = self.color_transform(image=img)["image"]

        return img, coords, was_flipped, rot_angle
