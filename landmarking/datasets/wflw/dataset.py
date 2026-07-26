"""WFLW dataset adapter with flip-aware and rotation augmentation.

Compatible with the preprocessed .pt format (affine-cropped 512×512 faces
with normalized [0,1] landmark coordinates).

Supports:
  - Horizontal flip with landmark reordering via WFLW_FLIP_PAIRS
  - Scale/translate affine augmentation
  - In-plane rotation augmentation (optional, via rot_factor > 0)
  - Color jitter (brightness, contrast, hue/sat)
"""

import math
import random

import cv2
import numpy as np
import torch
import albumentations as A

from ..base import BaseDataset
from .topology import FLIP_PERM_98, build_flip_permutation


class WFLWDataset(BaseDataset):
    """WFLW facial landmark dataset.

    Args:
        pt_paths: List of .pt file paths.
        input_size: Expected image size (square).
        num_landmarks: Number of landmarks (default 98).
        augment: Whether to apply augmentation (disable for val/test).
        flip_prob: Probability of horizontal flip when augment=True.
        rot_factor: Max rotation angle in degrees (0 = disabled).
        rot_prob: Probability of applying rotation when rot_factor > 0.
    """

    def __init__(
        self,
        pt_paths,
        input_size: int = 512,
        num_landmarks: int = 98,
        augment: bool = True,
        flip_prob: float = 0.5,
        rot_factor: float = 0,
        rot_prob: float = 0.6,
    ):
        self.paths = pt_paths
        self.input_size = input_size
        self.num_landmarks = num_landmarks
        self.augment = augment
        self.flip_prob = flip_prob
        self.rot_factor = rot_factor
        self.rot_prob = rot_prob

        # Use pre-built 98-point permutation or build one for other counts
        if num_landmarks == 98:
            self.flip_perm = FLIP_PERM_98
        else:
            self.flip_perm = np.arange(num_landmarks, dtype=np.int64)

        # Image-only augmentation (no landmark impact)
        self.color_transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3
            ),
        ])

        # Resize if input_size != native 512px
        self.resize_transform = None
        if input_size != 512:
            self.resize_transform = A.Compose([
                A.LongestMaxSize(max_size=input_size),
                A.PadIfNeeded(
                    input_size, input_size,
                    border_mode=cv2.BORDER_CONSTANT
                ),
            ])

        self.normalize = A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], weights_only=False)
        img = data["image"].permute(1, 2, 0).numpy()  # HWC uint8
        coords = data["tps"].numpy().copy()  # (N, 2) float32, already [0,1]
        orig_size = data.get(
            "orig_size", torch.tensor([img.shape[0], img.shape[1]])
        )

        was_flipped = False
        rot_angle = 0.0

        if self.augment:
            img, coords, was_flipped, rot_angle = self._augment(img, coords)

        # Resize to input_size if different from native 512
        if self.resize_transform is not None:
            resized = self.resize_transform(image=img)
            img = resized["image"]

        img_norm = self.normalize(image=img)["image"]
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        coords_tensor = torch.from_numpy(coords).float()

        metadata = {
            "orig_size": orig_size,
            "was_flipped": torch.tensor(was_flipped, dtype=torch.bool),
            "rot_angle": torch.tensor(rot_angle, dtype=torch.float32),
        }
        # Include attributes if available
        if "attrs" in data:
            metadata["attrs"] = data["attrs"]

        return img_tensor, coords_tensor, metadata

    def _augment(self, img: np.ndarray, coords: np.ndarray):
        """Apply augmentation. Returns (img, coords, was_flipped, rot_angle_deg)."""
        was_flipped = False
        rot_angle = 0.0

        # Horizontal flip
        if random.random() < self.flip_prob:
            img = img[:, ::-1, :].copy()
            coords[:, 0] = 1.0 - coords[:, 0]
            coords = coords[self.flip_perm]
            was_flipped = True

        # Affine (scale + translate)
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
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    coords = new_coords
                    break

        # In-plane rotation (optional)
        if self.rot_factor > 0 and random.random() < self.rot_prob:
            for _ in range(10):
                angle = random.uniform(-self.rot_factor, self.rot_factor)
                theta = math.radians(angle)
                cos_t, sin_t = math.cos(theta), math.sin(theta)
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
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    coords = new_coords
                    rot_angle = angle
                    break

        # Color jitter
        img = self.color_transform(image=img)["image"]

        return img, coords, was_flipped, rot_angle
