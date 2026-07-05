"""
WFLW dataset class — compatible with the existing LizardDataset .pt format
but supports any number of landmarks (not hardcoded to 9).

Drop-in replacement for LizardDataset when training on WFLW.
"""
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
import numpy as np


class WFLWDataset(Dataset):
    def __init__(self, pt_paths, input_size=512, num_landmarks=98):
        self.paths = pt_paths
        self.input_size = input_size
        self.num_landmarks = num_landmarks

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=input_size),
            A.PadIfNeeded(input_size, input_size, border_mode=cv2.BORDER_CONSTANT),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(0.05, 0.05),
                rotate=(-30, 30),
                p=0.8
            ),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10)
            ], p=0.7),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False, label_fields=[]))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx])
        img = data["image"].permute(1, 2, 0).numpy()   # HWC uint8
        coords = data["tps"].numpy()                    # (N, 2) float32

        H, W = img.shape[:2]
        coords[:, 0] = np.clip(coords[:, 0], 0, W - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, H - 1)
        keypoints = [(float(pt[0]), float(pt[1])) for pt in coords]

        orig_size = data.get("orig_size", torch.tensor([H, W]))

        for _ in range(10):
            augmented = self.transform(image=img, keypoints=keypoints)
            img_aug = augmented["image"]
            kp_aug = np.array(augmented["keypoints"], dtype=np.float32)
            if kp_aug.shape[0] == self.num_landmarks:
                break
        else:
            # Fallback: use original keypoints without augmentation
            img_aug = augmented["image"]
            kp_aug = np.array(keypoints, dtype=np.float32)

        kp_aug[:, 0] = np.clip(kp_aug[:, 0], 0, self.input_size - 1)
        kp_aug[:, 1] = np.clip(kp_aug[:, 1], 0, self.input_size - 1)
        coords_norm = kp_aug / self.input_size

        img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float()
        coords_tensor = torch.from_numpy(coords_norm).float()

        return img_tensor, coords_tensor, orig_size
