"""
WFLW dataset for HRNet heatmap regression.

Augmentation matches the paper (face_alignment_wflw_hrnet_w18.yaml):
  - Horizontal flip with WFLW 98-point flip pairs (p=0.5)
  - In-plane rotation: uniform ±30°, applied with p=0.6
  - Scale jitter: uniform ±25% around bbox scale
  - Brightness/contrast color jitter

Key difference from WFLWDataset (used by GCN):
  - Rotation is included — heatmap model has no mean-shape initialization
    so rotation does not require coordinating with an external prior.
  - Coordinates are transformed jointly with the image using cv2.warpAffine
    so the rotation is exactly consistent between image and landmarks.
  - The "reject if out of bounds" strategy is used for rotation (same as
    affine in WFLWDataset) to avoid clipping landmarks to image borders.

Returns (img_tensor, coords_tensor, orig_size, flipped_tensor) — same
4-tuple as WFLWDataset so it can be used interchangeably in training loops.
"""
import sys
import math
import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
import numpy as np

# Import WFLW_FLIP_PAIRS from graph_topology.py
_WFLW_DIR = Path(__file__).parent.parent.parent / "alternative-datasets" / "WFLW"
sys.path.insert(0, str(_WFLW_DIR))
from graph_topology import WFLW_FLIP_PAIRS


def _build_flip_permutation(flip_pairs, num_landmarks=98):
    perm = np.arange(num_landmarks, dtype=np.int64)
    for i, j in flip_pairs:
        perm[i] = j
        perm[j] = i
    return perm


_FLIP_PERM_98 = _build_flip_permutation(WFLW_FLIP_PAIRS, 98)


def _rotate_coords(coords, angle_deg, cx=0.5, cy=0.5):
    """Rotate (N, 2) coords in [0,1] space around centre (cx, cy).

    Args:
        coords:    (N, 2) float32 in [0, 1]
        angle_deg: rotation angle in degrees (positive = counter-clockwise)
        cx, cy:    rotation centre in [0,1] space (default: image centre)

    Returns:
        (N, 2) rotated coordinates
    """
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x = coords[:, 0] - cx
    y = coords[:, 1] - cy
    x_rot = cos_t * x - sin_t * y + cx
    y_rot = sin_t * x + cos_t * y + cy
    return np.stack([x_rot, y_rot], axis=-1).astype(np.float32)


class WFLWHeatmapDataset(Dataset):
    """WFLW dataset with full paper augmentation for heatmap regression."""

    def __init__(self, pt_paths, input_size=256, num_landmarks=98,
                 augment=True, flip_prob=0.5,
                 rot_factor=30, rot_prob=0.6,
                 scale_factor=0.25):
        self.paths = pt_paths
        self.input_size = input_size
        self.num_landmarks = num_landmarks
        self.augment = augment
        self.flip_prob = flip_prob
        self.rot_factor = rot_factor
        self.rot_prob = rot_prob
        self.scale_factor = scale_factor

        self.flip_perm = _FLIP_PERM_98 if num_landmarks == 98 else np.arange(num_landmarks)

        self.color_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15,
                                 val_shift_limit=10, p=0.3),
        ])
        self.normalize = A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx])
        img = data["image"].permute(1, 2, 0).numpy()   # HWC uint8, 512×512
        coords = data["tps"].numpy().copy()             # (N,2) float32 [0,1]
        orig_size = data.get("orig_size", torch.tensor([img.shape[0], img.shape[1]]))

        # Downscale from native 512px to input_size if needed
        if img.shape[0] != self.input_size:
            img = cv2.resize(img, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_LINEAR)

        was_flipped = False
        if self.augment:
            img, coords, was_flipped = self._augment(img, coords)

        img_norm = self.normalize(image=img)["image"]
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        coords_tensor = torch.from_numpy(coords).float()
        return img_tensor, coords_tensor, orig_size, torch.tensor(was_flipped, dtype=torch.bool)

    def _augment(self, img, coords):
        was_flipped = False

        # ── Horizontal flip ───────────────────────────────────────────────
        if random.random() < self.flip_prob:
            img = img[:, ::-1, :].copy()
            coords[:, 0] = 1.0 - coords[:, 0]
            coords = coords[self.flip_perm]
            was_flipped = True

        # ── Scale jitter ──────────────────────────────────────────────────
        # Apply before rotation so the face fills a consistent region
        if random.random() < 0.7:
            for _ in range(10):
                scale = random.uniform(1 - self.scale_factor, 1 + self.scale_factor)
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
                    img = cv2.warpAffine(img, M, (self.input_size, self.input_size),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    coords = new_coords
                    break

        # ── In-plane rotation ─────────────────────────────────────────────
        # Applied with rot_prob, angle uniform in [-rot_factor, rot_factor].
        # Uses "reject if out of bounds" — if rotated landmarks leave [0,1]
        # we skip rotation rather than clipping, which would collapse landmarks
        # to image borders.
        if random.random() < self.rot_prob:
            for _ in range(10):
                angle = random.uniform(-self.rot_factor, self.rot_factor)
                new_coords = _rotate_coords(coords, angle)
                if new_coords.min() >= 0.0 and new_coords.max() <= 1.0:
                    cx, cy = self.input_size / 2, self.input_size / 2
                    M = cv2.getRotationMatrix2D((cx, cy), -angle,  # cv2 is CW positive
                                               1.0)
                    img = cv2.warpAffine(img, M, (self.input_size, self.input_size),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    coords = new_coords
                    break
            # If no valid angle found, skip rotation (face likely near border)

        # ── Color jitter ──────────────────────────────────────────────────
        img = self.color_transform(image=img)["image"]

        return img, coords, was_flipped
