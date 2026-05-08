import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torch.nn.functional as F


class ViTDataset(Dataset):
    def __init__(self, source):
        if isinstance(source, list):
            self.files = [Path(p) for p in source]
        else:
            self.folder = Path(source)
            self.files = sorted(self.folder.glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        img = data["image"]  # (3, 512, 512) float32, pre-normalized

        # Resize from 512 to 224 (bilinear, keeps normalization intact)
        img_224 = F.interpolate(
            img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze(0)

        # Keypoints are in 512-space pixel coords, normalize to [0,1]
        keypoints = data["tps"].float() / 512.0

        return img_224, keypoints
