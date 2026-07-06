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

        # Affine augmentation: scale + translate only (no rotation).
        # albumentations handles keypoint remapping for these ops automatically
        # since they don't change the semantic identity of landmarks.
        self.affine_transform = A.Compose([
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=0,
                p=0.7,
            ),
        ], keypoint_params=A.KeypointParams(
            format="xy", remove_invisible=False, label_fields=[]
        ))

        # Image-only augmentation (no landmark impact)
        self.color_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
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

        if self.augment:
            img, coords = self._augment(img, coords)

        img_norm = self.normalize(image=img)["image"]
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        coords_tensor = torch.from_numpy(coords).float()

        return img_tensor, coords_tensor, orig_size

    def _augment(self, img: np.ndarray, coords: np.ndarray):
        """Apply augmentation to image and landmarks.

        Handles flip separately from affine because flip requires landmark
        reordering that albumentations cannot do automatically.
        """
        # ── Horizontal flip ───────────────────────────────────────────────
        if random.random() < self.flip_prob:
            img = img[:, ::-1, :].copy()          # mirror image horizontally
            coords[:, 0] = 1.0 - coords[:, 0]    # mirror x coordinates
            coords = coords[self.flip_perm]        # reorder landmarks

        # ── Affine (scale + translate) ────────────────────────────────────
        # Convert [0,1] coords to pixel space for albumentations
        kps_px = [(float(x * self.input_size), float(y * self.input_size))
                  for x, y in coords]

        for _ in range(5):  # retry if any keypoints go out of bounds
            result = self.affine_transform(image=img, keypoints=kps_px)
            kps_out = np.array(result["keypoints"], dtype=np.float32)
            if kps_out.shape[0] == self.num_landmarks:
                img = result["image"]
                # Clip and renormalise back to [0,1]
                kps_out[:, 0] = np.clip(kps_out[:, 0], 0, self.input_size - 1)
                kps_out[:, 1] = np.clip(kps_out[:, 1], 0, self.input_size - 1)
                coords = kps_out / self.input_size
                break
        # If all retries lost keypoints, coords stays as-is (pre-affine)

        # ── Color jitter ──────────────────────────────────────────────────
        img = self.color_transform(image=img)["image"]

        return img, coords
