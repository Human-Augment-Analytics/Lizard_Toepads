"""Model registry and model implementations.

Importing this package triggers registration of all model variants.
Models that require optional dependencies (torch_geometric) are imported
conditionally — they are registered only if the dependency is available.
"""

import importlib
import warnings

# Import registry first
from .registry import MODEL_REGISTRY, get_model, register_model

# List of model modules to import for registration
_MODEL_MODULES = [
    "hrnet_gcn",              # "standard" — requires torch_geometric
    "hrnet_gcn_ms",           # "multiscale" — requires torch_geometric
    "hrnet_gcn_coord",        # "coord" — requires torch_geometric
    "hrnet_gcn_fused",        # "fused" — requires torch_geometric
    "hrnet_gcn_fused_global", # "fused_global" — requires torch_geometric
    "hrnet_gcn_fused_global_ms", # "fused_global_ms" — requires torch_geometric
    "hrnet_gcn_fused_global_patch", # "fused_global_patch" — requires torch_geometric
    "hrnet_gcn_hinit",        # "hinit" — requires torch_geometric
    "graph_cond_heatmap",     # "graph_cond_heatmap" — requires torch_geometric
    "graph_prior_fusion",     # "graph_prior_fusion" — requires torch_geometric
    "star_head",              # "fused_global_star" — requires torch_geometric
    "hrnet_heatmap",          # "heatmap"
    "hrnet_coord",            # "hrnet_coord"
    "stacked_hourglass",      # "stacked_hourglass"
    "vit",                    # "vit"
    "pipnet",                 # "pipnet"
]

for _mod_name in _MODEL_MODULES:
    try:
        importlib.import_module(f".{_mod_name}", package=__name__)
    except ImportError as e:
        warnings.warn(
            f"Could not import landmarking.models.{_mod_name}: {e}. "
            f"Some model variants may not be available.",
            stacklevel=2,
        )

__all__ = ["MODEL_REGISTRY", "get_model", "register_model"]
