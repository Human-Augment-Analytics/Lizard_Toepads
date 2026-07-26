"""Model registry for landmark detection models.

Provides a simple dict-based registry with a decorator for registration
and a helper function for model instantiation by variant name.
"""

from typing import Dict, Type


MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str):
    """Decorator to register a model class in the global registry.

    Args:
        name: The variant key to register the model under.

    Returns:
        Decorator function that registers and returns the class unchanged.
    """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(variant: str, **kwargs):
    """Instantiate a model by variant name.

    Args:
        variant: Registry key identifying the model architecture.
        **kwargs: Arguments passed to the model constructor.

    Returns:
        An instantiated model.

    Raises:
        KeyError: If variant is not found in the registry.
    """
    if variant not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model variant '{variant}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[variant](**kwargs)
