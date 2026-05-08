import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENORMALIZE = A.Compose(
    [
        A.LongestMaxSize(max_size=224),
        A.PadIfNeeded(
            min_height=224,
            min_width=224,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(
        format="xy",
        remove_invisible=False
    )
)


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
        img = data["image"].permute(1, 2, 0).numpy()
        keypoints = data["tps"].numpy()

        aug = IMAGENORMALIZE(image=img, keypoints=keypoints.tolist())
        img_tensor = aug["image"]
        kps = np.array(aug["keypoints"], dtype=np.float32) / 224.0

        return img_tensor, torch.tensor(kps, dtype=torch.float32)
