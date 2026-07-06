"""
WFLW dataset class — compatible with the existing LizardDataset .pt format
but supports any number of landmarks (not hardcoded to 9).

Drop-in replacement for LizardDataset when training on WFLW.

Coordinate convention:
  preprocess.py saves "tps" as (N, 2) float32 in [0, 1], normalized to the
  512x512 letterboxed image. This class reads them directly — no re-scaling.
"""
import torch
from torch.utils.data import Dataset
import albumentations as A
import numpy as np


class WFLWDataset(Dataset):
    def __init__(self, pt_paths, input_size=512, num_landmarks=98):
        self.paths = pt_paths
        self.input_size = input_size
        self.num_landmarks = num_landmarks

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx])
        img = data["image"].permute(1, 2, 0).numpy()   # HWC uint8, already 512x512
        coords_norm = data["tps"].numpy()               # (N, 2) float32, already [0,1]

        orig_size = data.get("orig_size", torch.tensor([img.shape[0], img.shape[1]]))

        # Image-only augmentation — no geometry transforms, so coords are unaffected.
        # The .pt files are already letterbox-cropped to input_size x input_size by
        # preprocess.py, so no resize/pad is needed here either.
        img_only_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        img_aug = img_only_transform(image=img)["image"]

        img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float()
        coords_tensor = torch.from_numpy(coords_norm).float()

        return img_tensor, coords_tensor, orig_size
