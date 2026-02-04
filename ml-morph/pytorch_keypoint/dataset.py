import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None

@dataclass
class Annotation:
    image_path: Path
    bbox: Tuple[int, int, int, int]  # left, top, width, height
    keypoints: np.ndarray  # shape: (num_keypoints, 2)


def _parse_xml(xml_path: Path, root_dir: Path) -> List[Annotation]:
    tree = ET.parse(xml_path)
    annotations: List[Annotation] = []

    for image_node in tree.findall(".//image"):
        relative_path = image_node.get("file")
        if not relative_path:
            continue

        image_path = (xml_path.parent / relative_path).resolve()
        if not image_path.exists() and root_dir:
            candidate = (root_dir / relative_path).resolve()
            if candidate.exists():
                image_path = candidate

        # Parse bounding boxes and keypoints
        for box in image_node.findall("box"):
            left = int(box.get("left", 0))
            top = int(box.get("top", 0))
            width = int(box.get("width", 0))
            height = int(box.get("height", 0))

            parts = box.findall("part")
            keypoints = np.zeros((len(parts), 2), dtype=np.float32)
            for idx, part in enumerate(parts):
                x = float(part.get("x", 0))
                y = float(part.get("y", 0))
                keypoints[idx] = (x, y)

            annotations.append(
                Annotation(
                    image_path=image_path,
                    bbox=(left, top, width, height),
                    keypoints=keypoints,
                )
            )
    return annotations


class ToepadKeypointDataset(Dataset):
    """
    Dataset that reads dlib-style XML annotations and returns normalized keypoints.
    """

    def __init__(
        self,
        xml_path: str,
        root_dir: str,
        image_size: int = 256,
        augment: bool = False,
        return_metadata: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.xml_path = Path(xml_path).resolve()
        self.annotations = _parse_xml(self.xml_path, self.root_dir)
        if not self.annotations:
            raise ValueError(f"No annotations found in {xml_path}")

        self.image_size = image_size
        self.augment = augment
        self.num_keypoints = self.annotations[0].keypoints.shape[0]
        self.return_metadata = return_metadata

        base_transforms = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]

        if self.augment:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
                ]
                + base_transforms
            )
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        ann = self.annotations[idx]
        image = Image.open(ann.image_path).convert("RGB")

        left, top, width, height = ann.bbox
        right = left + width
        bottom = top + height

        # Clamp crop to image bounds
        image_w, image_h = image.size
        left = max(0, left)
        top = max(0, top)
        right = min(image_w, right)
        bottom = min(image_h, bottom)

        cropped = image.crop((left, top, right, bottom))

        # Normalize keypoints relative to crop
        keypoints = ann.keypoints.copy()
        keypoints[:, 0] = (keypoints[:, 0] - left) / max(1.0, right - left)
        keypoints[:, 1] = (keypoints[:, 1] - top) / max(1.0, bottom - top)
        keypoints = np.clip(keypoints, 0.0, 1.0)

        if keypoints.shape[0] != self.num_keypoints:
            adjusted = np.zeros((self.num_keypoints, 2), dtype=np.float32)
            count = min(self.num_keypoints, keypoints.shape[0])
            adjusted[:count] = keypoints[:count]
            keypoints = adjusted

        tensor_image = self.transform(cropped)
        keypoint_tensor = torch.from_numpy(keypoints).float()

        sample = {
            "image": tensor_image,
            "keypoints": keypoint_tensor.view(-1),
        }

        if self.return_metadata:
            sample["metadata"] = {
                "image_path": str(ann.image_path),
                "bbox": (left, top, right - left, bottom - top),
                "original_size": (image_w, image_h),
                "crop_size": (right - left, bottom - top),
            }

        return sample

