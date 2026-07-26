"""Unified Lizard dataset adapter supporting both coord and heatmap modes.

Loads preprocessed .pt crop files (OBB-cropped, letterboxed to 512×512)
and returns ImageNet-normalized images with landmark coordinates.

Coord mode:   Returns (img, coords, metadata)
Heatmap mode: Returns (img, coords, heatmaps, metadata)
"""

import numpy as np
import cv2
import torch
import albumentations as A

from ..base import BaseDataset
from ...common.heatmap_utils import generate_gaussian_heatmap


class LizardDataset(BaseDataset):
    """Lizard toe-pad landmark dataset.

    Loads .pt files containing OBB-cropped, letterbox-padded images
    with 9 landmarks in pixel space (already in 512×512 canvas).

    Args:
        pt_paths: List of paths to .pt crop files.
        input_size: Expected image canvas size (square).
        num_landmarks: Number of landmarks (default 9).
        augment: Whether to apply geometric+color augmentation.
        mode: "coord" for coordinate regression, "heatmap" for heatmap mode.
        heatmap_size: Output heatmap resolution (only used in heatmap mode).
        sigma: Gaussian kernel sigma (only used in heatmap mode).
    """

    def __init__(
        self,
        pt_paths,
        input_size: int = 512,
        num_landmarks: int = 9,
        augment: bool = True,
        mode: str = "coord",
        heatmap_size: int = 128,
        sigma: float = 4.0,
    ):
        self.paths = pt_paths
        self.input_size = input_size
        self.num_landmarks = num_landmarks
        self.augment = augment
        self.mode = mode
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        # Augmentation pipeline with keypoint awareness
        if augment:
            self.transform = A.Compose(
                [
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
                    A.GaussNoise(var_limit=(1, 5), p=0.3),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ],
                keypoint_params=A.KeypointParams(
                    format="xy", remove_invisible=False
                ),
            )
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ],
                keypoint_params=A.KeypointParams(
                    format="xy", remove_invisible=False
                ),
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], weights_only=False)
        img = data["image"].permute(1, 2, 0).numpy()  # CHW uint8 → HWC
        coords = data["tps"].numpy()  # (9, 2) float32 — pixel coords in 512 space

        # Clamp coordinates to image bounds
        H, W = img.shape[:2]
        coords[:, 0] = np.clip(coords[:, 0], 0, W - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, H - 1)
        keypoints = coords.tolist()

        # Apply augmentation with retry on keypoint dropout
        for _ in range(10):
            augmented = self.transform(image=img, keypoints=keypoints)
            img_aug = augmented["image"]
            kp_aug = np.array(augmented["keypoints"], dtype=np.float32)
            if kp_aug.shape[0] == self.num_landmarks:
                break
        else:
            # Fallback: use last result or raw keypoints
            img_aug = augmented["image"]
            kp_aug = np.array(keypoints, dtype=np.float32)

        # Clamp augmented keypoints and normalize to [0, 1]
        kp_aug[:, 0] = np.clip(kp_aug[:, 0], 0, self.input_size - 1)
        kp_aug[:, 1] = np.clip(kp_aug[:, 1], 0, self.input_size - 1)
        coords_norm = kp_aug / self.input_size

        img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float()
        coords_tensor = torch.from_numpy(coords_norm).float()

        # Build metadata dict
        metadata = {
            "orig_size": data.get(
                "orig_size", torch.tensor([H, W], dtype=torch.float32)
            ),
        }
        # Include back-projection info if available
        if "M" in data:
            metadata["M"] = data["M"]
        if "scale" in data:
            metadata["scale"] = data["scale"]
        if "pad" in data:
            metadata["pad"] = data["pad"]
        if "class_name" in data:
            metadata["class_name"] = data["class_name"]
        if "ruler_px" in data:
            metadata["ruler_px"] = data["ruler_px"]

        if self.mode == "heatmap":
            heatmaps = generate_gaussian_heatmap(
                coords_norm, self.heatmap_size, self.sigma
            )
            heatmaps_tensor = torch.from_numpy(heatmaps).float()
            return img_tensor, coords_tensor, heatmaps_tensor, metadata

        return img_tensor, coords_tensor, metadata
