import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import albumentations as A
import cv2

class LizardDataset(torch.utils.data.Dataset):
    def __init__(self, npz_paths, aug_factor=8):
        self.paths = npz_paths
        self.aug_factor = aug_factor
        self.heatmap_size = 128
    def __len__(self):
        return len(self.paths) * self.aug_factor

    def __getitem__(self, idx):
        true_idx = idx % len(self.paths)
        data = torch.load(self.paths[true_idx])
        img = data["image"].permute(1, 2, 0).numpy()
        heatmaps = data["heatmap"].permute(1, 2, 0).numpy()
        img, heatmaps = apply_augmentation(img, heatmaps)

        img_tensor = torch.from_numpy(img).float()
        heatmaps_tensor = torch.from_numpy(heatmaps).permute(2,0,1).float()
        heatmaps_tensor = F.interpolate(
            heatmaps_tensor.unsqueeze(0),
            size=(128,128),
            mode='area',
        )
        heatmaps_tensor = heatmaps_tensor.squeeze(0)

        heatmaps_tensor = heatmaps_tensor / (heatmaps_tensor.max() + 1e-8)
        return img_tensor, heatmaps_tensor


def apply_augmentation(img, heatmap):
    aug_transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5)
        ], p=0.7),
        A.GaussNoise(var_limit=(5, 20), p=0.3),
    ], additional_targets={"heatmap": "mask"})

    out = aug_transform(image=img, mask=heatmap)
    image = out["image"]
    heatmap = out["mask"]

    return image, heatmap
