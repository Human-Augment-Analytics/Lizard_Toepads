"""Unit tests for model registry.

Tests verify registry dict contents and KeyError behavior.
Model instantiation is NOT tested here because it requires
timm/torch_geometric which may not be available or requires GPU.

Models that require torch_geometric are conditionally imported.
If torch_geometric is not installed, only non-GCN models will be
in the registry. Tests account for this.
"""

import pytest

from landmarking.models.registry import MODEL_REGISTRY, get_model


# All variant keys from the design document
ALL_VARIANTS = [
    "standard",
    "multiscale",
    "coord",
    "fused",
    "fused_global",
    "hinit",
    "graph_cond_heatmap",
    "heatmap",
    "hrnet_coord",
    "stacked_hourglass",
    "vit",
]

# Variants that don't require torch_geometric
CORE_VARIANTS = ["heatmap", "hrnet_coord", "stacked_hourglass", "vit"]

# Variants that require torch_geometric
GCN_VARIANTS = ["standard", "multiscale", "coord", "fused", "fused_global", "hinit", "graph_cond_heatmap"]


def _has_torch_geometric():
    try:
        import torch_geometric  # noqa: F401
        return True
    except ImportError:
        return False


class TestModelRegistry:
    """Test that the registry is correctly populated."""

    def test_core_variants_always_registered(self):
        """Core variants (no torch_geometric) should always be present."""
        import landmarking.models  # noqa: F401

        for variant in CORE_VARIANTS:
            assert variant in MODEL_REGISTRY, (
                f"Core variant '{variant}' not found in registry. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

    @pytest.mark.skipif(
        not _has_torch_geometric(),
        reason="torch_geometric not installed"
    )
    def test_all_10_variants_registered_with_torch_geometric(self):
        """All 11 variant keys should be in the registry when deps available."""
        import landmarking.models  # noqa: F401

        for variant in ALL_VARIANTS:
            assert variant in MODEL_REGISTRY

    def test_registry_values_are_classes(self):
        import landmarking.models  # noqa: F401
        for name, cls in MODEL_REGISTRY.items():
            assert isinstance(cls, type), (
                f"Registry value for '{name}' is not a class: {cls}"
            )

    def test_get_model_raises_keyerror_on_invalid(self):
        """get_model should raise KeyError for unknown variants."""
        import landmarking.models  # noqa: F401

        with pytest.raises(KeyError, match="Unknown model variant"):
            get_model("nonexistent_model", num_landmarks=9)

    def test_get_model_error_message_lists_available(self):
        """KeyError message should list available variants."""
        import landmarking.models  # noqa: F401

        with pytest.raises(KeyError) as exc_info:
            get_model("bad_variant", num_landmarks=9)

        error_msg = str(exc_info.value)
        assert "Available:" in error_msg

    def test_variant_keys_are_strings(self):
        import landmarking.models  # noqa: F401
        for key in MODEL_REGISTRY:
            assert isinstance(key, str)

    def test_at_least_core_variants_count(self):
        """At minimum, the 4 core variants should be present."""
        import landmarking.models  # noqa: F401
        assert len(MODEL_REGISTRY) >= len(CORE_VARIANTS)
