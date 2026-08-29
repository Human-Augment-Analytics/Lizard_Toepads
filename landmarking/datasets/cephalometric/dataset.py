"""Cephalometric (ISBI 2015) dataset adapter supporting coord and heatmap modes.

Loads preprocessed .pt files (grayscale radiographs replicated to 3 channels,
with 19 landmark coordinates already normalized to [0, 1]) and returns
ImageNet-normalized images with landmark coordinates.

Coord mode:   Returns (img, coords, metadata)
Heatmap mode: Returns (img, coords, heatmaps, metadata)

Every radiograph is a single sample with exactly 19 always-present landmarks:
there is no bounding-box cropping, multi-instance handling, or visibility
masking (mirrors WFLW's [0,1] tps consumption and Lizard's heatmap mode).
"""

import cv2
import numpy as np
import torch
import albumentations as A

from ..base import BaseDataset
from ...common.heatmap_utils import generate_gaussian_heatmap


class CephalometricDataset(BaseDataset):
    """ISBI 2015 cephalometric X-ray landmark dataset.

    Loads .pt files containing a ``(3, H, W)`` uint8 grayscale-replicated image
    and a ``(19, 2)`` float32 ``tps`` tensor already normalized to ``[0, 1]``.

    Args:
        pt_paths: List of paths to preprocessed .pt files.
        input_size: Square image canvas size for model input.
        num_landmarks: Number of landmarks (default 19).
        augment: Whether to apply light image augmentation (never masking).
        mode: "coord" for coordinate regression, "heatmap" for heatmap mode.
        heatmap_size: Output heatmap resolution (heatmap mode only).
        sigma: Gaussian kernel sigma (heatmap mode only).
        pixel_spacing: mm per pixel used when the .pt omits it (ISBI default 0.1).
        landmark_indices: Optional sparsity subset into 0..18.
        split: Partition identifier ("train" | "test1" | "test2").
    """

    def __init__(
        self,
        pt_paths,
        input_size: int = 512,
        num_landmarks: int = 19,
        augment: bool = False,
        mode: str = "coord",
        heatmap_size: int = 128,
        sigma: float = 4.0,
        pixel_spacing: float = 0.1,
        landmark_indices: list = None,
        split: str = "train",
    ):
        self.paths = pt_paths
        self.input_size = input_size
        self.num_landmarks = num_landmarks
        self.augment = augment
        self.mode = mode
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.pixel_spacing = pixel_spacing
        self.landmark_indices = landmark_indices or []
        self.split = split

        # Conditional resize only when input_size differs from native 512
        # canvas (mirrors WFLWDataset).
        self.resize_transform = None
        if input_size != 512:
            self.resize_transform = A.Compose([
                A.LongestMaxSize(max_size=input_size),
                A.PadIfNeeded(
                    input_size, input_size,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ])

        # Optional light color augmentation (image-only, never affects
        # landmark positions and never introduces masking).
        self.color_transform = None
        if augment:
            self.color_transform = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
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

        img = data["image"]
        # Defensively replicate a single-channel image to 3 channels.
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img = img.permute(1, 2, 0).numpy()  # CHW uint8 -> HWC

        coords = data["tps"].numpy().copy()  # (19, 2) float32, already [0,1]

        # Slice coordinates to subset when landmark_indices is active
        # (matches WFLWDataset's `coords[self.landmark_indices]` semantics).
        if self.landmark_indices:
            coords = coords[self.landmark_indices]

        # Optional light color augmentation (image-only).
        if self.color_transform is not None:
            img = self.color_transform(image=img)["image"]

        # Resize to input_size only when different from native 512.
        if self.resize_transform is not None:
            img = self.resize_transform(image=img)["image"]

        img_norm = self.normalize(image=img)["image"]
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        coords_tensor = torch.from_numpy(coords).float()

        H, W = img.shape[:2]
        metadata = {
            "orig_size": data.get(
                "orig_size", torch.tensor([H, W], dtype=torch.float32)
            ),
            "pixel_spacing": torch.tensor(
                float(data.get("pixel_spacing", self.pixel_spacing)),
                dtype=torch.float32,
            ),
            "split": data.get("split", self.split),
        }

        if self.mode == "heatmap":
            heatmaps = generate_gaussian_heatmap(
                coords, self.heatmap_size, self.sigma
            )
            heatmaps_tensor = torch.from_numpy(heatmaps).float()
            return img_tensor, coords_tensor, heatmaps_tensor, metadata

        return img_tensor, coords_tensor, metadata
