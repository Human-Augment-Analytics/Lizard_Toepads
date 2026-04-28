import torch
from torch.utils.data import Dataset
from pathlib import Path

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
        img = data["image"].float()
        keypoints = data["keypoints"].float()
        keypoints = keypoints / 224

        return img, keypoints
