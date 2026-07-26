"""Base dataset class for landmark detection.

Defines the abstract interface that all dataset adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for all landmark datasets.

    All datasets return a tuple of:
        img_tensor: (3, H, W) float32, ImageNet-normalized
        coords:     (num_landmarks, 2) float32, in [0, 1]
        metadata:   dict with at minimum {"orig_size": tensor}

    Subclasses may return additional elements (e.g., heatmaps) after
    coords but before metadata for mode-specific outputs.
    """

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Load and return one sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (img_tensor, coords, metadata) at minimum.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...
