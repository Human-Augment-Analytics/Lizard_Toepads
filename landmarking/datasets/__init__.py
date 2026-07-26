"""Dataset adapters for landmark detection."""

from .base import BaseDataset
from .lizard import LizardDataset
from .wflw import WFLWDataset

__all__ = ["BaseDataset", "LizardDataset", "WFLWDataset"]
