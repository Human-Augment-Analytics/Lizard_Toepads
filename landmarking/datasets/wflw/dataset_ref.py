"""WFLW Reference Dataset — paper-faithful HRNet heatmap pipeline.

Ported from `alternative-models/hrnet/wflw_pt_dataset.py`. Uses local
`ref_transforms` module (no sys.path hacks to external repos).

Returns: (img_tensor, target_hm, meta)
  img_tensor: (3, 256, 256) float32, ImageNet-normalised
  target_hm:  (98, 64, 64) float32, 3σ truncated Gaussian heatmap targets
  meta:       dict with index, center (Tensor[2]), scale (float),
              pts (Tensor[98,2] in 512px space), tpts (Tensor[98,2] in 64px space)
"""

import random

import cv2
import numpy as np
import torch
import torch.utils.data as data

from .ref_transforms import (
    crop_v2,
    fliplr_joints,
    generate_target,
    get_affine_transform,
    transform_pixel,
)


# ── Constants ─────────────────────────────────────────────────────────────────

_FIXED_CENTER = torch.Tensor([256.0, 256.0])
_FIXED_SCALE = 512.0 / 200.0  # = 2.56

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_INPUT_SIZE = [256, 256]
_HEATMAP_SIZE = [64, 64]
_SIGMA = 1.5
_NUM_LANDMARKS = 98


class WFLWRefDataset(data.Dataset):
    """WFLW dataset with reference HRNet augmentation pipeline.

    Reads pre-cropped 512×512 .pt files and applies the same augmentation
    as the official HRNet-Facial-Landmark-Detection paper:
      - Horizontal flip with landmark pair swapping (p=0.5)
      - Scale jitter ±25%
      - Rotation ±30° with probability 0.6
      - crop_v2 (cv2 affine) to 256×256
      - generate_target for 3σ truncated Gaussian heatmaps on 64×64

    Args:
        pt_paths: List of .pt file path strings.
        augment: Whether to apply augmentation (True for train).
        flip_prob: Probability of horizontal flip.
        scale_factor: Scale jitter range (±).
        rot_factor: Max rotation angle in degrees.
    """

    def __init__(
        self,
        pt_paths,
        augment=True,
        flip_prob=0.5,
        scale_factor=0.25,
        rot_factor=30,
    ):
        self.paths = list(pt_paths)
        self.augment = augment
        self.flip_prob = flip_prob
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pt_path = self.paths[idx]

        # Step 1: Load .pt file
        data_dict = torch.load(pt_path, map_location="cpu", weights_only=False)

        # img: (3, 512, 512) uint8 tensor → (512, 512, 3) float32 HWC
        img = data_dict["image"].permute(1, 2, 0).numpy().astype(np.float32)

        # tps: (98, 2) float32 [0,1] → 512px pixel space
        tps = data_dict["tps"].numpy().astype(np.float64)
        pts = tps * 512.0  # (98, 2) in pixel space

        # Step 2: Set fixed center and scale
        center = _FIXED_CENTER.clone()
        scale = _FIXED_SCALE

        # Step 3: Augmentation
        r = 0  # rotation angle (degrees)

        if self.augment:
            # Horizontal flip (p=flip_prob)
            if random.random() <= self.flip_prob:
                img = np.fliplr(img).copy()
                pts = fliplr_joints(pts, width=512, dataset='WFLW')
                center[0] = 512 - center[0]

            # Scale jitter
            scale = scale * random.uniform(
                1 - self.scale_factor, 1 + self.scale_factor
            )

            # Rotation (p=0.6, matching paper)
            if random.random() <= 0.6:
                r = random.uniform(-self.rot_factor, self.rot_factor)

        # Step 4: Crop to 256×256 using crop_v2 (cv2-only affine)
        # Convert center to numpy for get_affine_transform
        center_np = center.numpy().astype(np.float32)
        scale_np = np.array([scale, scale], dtype=np.float32)

        img_cropped = crop_v2(
            img.astype(np.uint8), center_np, scale_np, _INPUT_SIZE, rot=r
        )

        # Step 5: Transform landmarks to 64px heatmap space
        tpts = pts.copy()
        for i in range(_NUM_LANDMARKS):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(
                    tpts[i, 0:2] + 1,  # +1: convert to 1-indexed (reference convention)
                    center_np,
                    scale,
                    _HEATMAP_SIZE,
                    rot=r,
                )

        # Step 6: Generate Gaussian heatmap targets (3σ truncated)
        target = np.zeros(
            (_NUM_LANDMARKS, _HEATMAP_SIZE[0], _HEATMAP_SIZE[1]),
            dtype=np.float32,
        )
        for i in range(_NUM_LANDMARKS):
            if tpts[i, 1] > 0:
                target[i] = generate_target(
                    target[i],
                    tpts[i] - 1,  # -1: back to 0-indexed for generate_target
                    _SIGMA,
                )

        # Step 7: Normalise image
        img_norm = img_cropped.astype(np.float32) / 255.0
        img_norm = (img_norm - _IMAGENET_MEAN) / _IMAGENET_STD
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        # Step 8: Build meta dict
        meta = {
            "index": idx,
            "center": center,       # Tensor[2], post-flip
            "scale": scale,         # float, post-jitter
            "pts": torch.Tensor(pts),    # (98,2) 512px, post-flip
            "tpts": torch.Tensor(tpts),  # (98,2) 64px heatmap space
        }

        return img_tensor, torch.Tensor(target), meta
