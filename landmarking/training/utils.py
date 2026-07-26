"""Training utilities for the landmarking package.

Provides seed setting, device resolution, and parameter group helpers.
"""

import os
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all relevant libraries.

    Sets seeds for: Python random, NumPy, PyTorch CPU/CUDA.
    Also configures CUDA for deterministic behavior.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cuda") -> torch.device:
    """Resolve a device string to a torch.device.

    Falls back to CPU if CUDA is requested but not available.

    Args:
        device_str: Device specification (e.g., "cuda", "cuda:0", "cpu").

    Returns:
        Resolved torch.device.
    """
    if "cuda" in device_str and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def make_param_groups(model, lr: float, lr_backbone: float) -> list:
    """Create differential learning rate parameter groups.

    Separates model parameters into:
      - backbone: trained with lr_backbone (fine-tuning, slower)
      - head: trained with lr (from scratch, faster)

    The backbone is identified by the 'backbone' attribute on the model.
    All other parameters are considered head parameters.

    Args:
        model: PyTorch model with a .backbone attribute.
        lr: Learning rate for the head (GCN/MLP) parameters.
        lr_backbone: Learning rate for the backbone parameters.

    Returns:
        List of parameter group dicts suitable for torch.optim.Adam.
    """
    if hasattr(model, "backbone"):
        backbone_params = list(model.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    else:
        # No backbone attribute — all params get head LR
        backbone_params = []
        head_params = list(model.parameters())

    groups = []
    if backbone_params:
        groups.append({
            "params": backbone_params,
            "lr": lr_backbone,
            "weight_decay": 0.0,
        })
    groups.append({
        "params": head_params,
        "lr": lr,
        "weight_decay": 0.0,
    })
    return groups


def make_output_dir(output_root: str, dataset_name: str, variant: str, run_id: str = None) -> str:
    """Generate a unique output directory path.

    Format: {output_root}/{dataset_name}/{variant}/{run_id}

    Args:
        output_root: Root output directory.
        dataset_name: Dataset name (e.g., "lizard", "wflw").
        variant: Model variant name.
        run_id: Optional run identifier. If None, uses timestamp.

    Returns:
        String path to the output directory (not yet created).
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_root, dataset_name, variant, run_id)
