import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class LizardDataset(Dataset):
    def __init__(self, pt_paths, input_size=512, heatmap_size=512):
        self.paths = pt_paths
        self.input_size = input_size
        self.heatmap_size = heatmap_size

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=input_size),
            A.PadIfNeeded(input_size, input_size, border_mode=cv2.BORDER_CONSTANT),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.05, 0.05),
                rotate=(-25, 25),
                p=0.8
            ),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10)
            ], p=0.7),
            #A.ElasticTransform(alpha=10, sigma=5, p=0.2),
            A.GaussNoise(var_limit=(1,5), p=0.3),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False, label_fields=[]))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx])
        img = data["image"].permute(1,2,0).numpy()
        coords = data["tps"].numpy()  # shape (9,2)

        # Clip coords inside original image bounds
        H, W = img.shape[:2]
        coords[:,0] = np.clip(coords[:,0], 0, W-1)
        coords[:,1] = np.clip(coords[:,1], 0, H-1)
        keypoints = [(float(pt[0]), float(pt[1])) for pt in coords]
        #print(data)
        orig_size = data["orig_size"]
        augmented = self.transform(image=img, keypoints=keypoints)
        
        img_aug = augmented["image"]
        kp_aug = np.array(augmented["keypoints"], dtype=np.float32)
        #print(kp_aug.shape)
        if kp_aug.shape[0] != 9:
            raise ValueError(f"Augmented keypoints shape mismatch: {kp_aug.shape}, expected 9")

        # Clamp and normalize
        kp_aug[:,0] = np.clip(kp_aug[:,0], 0, self.input_size-1)
        kp_aug[:,1] = np.clip(kp_aug[:,1], 0, self.input_size-1)
        coords_norm = kp_aug / self.input_size

        img_tensor = torch.from_numpy(img_aug).permute(2,0,1).float()
        coords_tensor = torch.from_numpy(coords_norm).float()

        return img_tensor, coords_tensor, orig_size