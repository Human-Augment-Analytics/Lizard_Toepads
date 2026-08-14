"""Unit tests for GraphCondHeatmap model basics."""

import pytest
import torch

from landmarking.models.graph_cond_heatmap import GraphCondHeatmap
from landmarking.models.registry import MODEL_REGISTRY
from landmarking.common.graph_topologies import make_chain_edge_index


def test_model_registered():
    """graph_cond_heatmap is registered in the MODEL_REGISTRY."""
    # Force import to trigger registration
    import landmarking.models.graph_cond_heatmap  # noqa: F401
    assert "graph_cond_heatmap" in MODEL_REGISTRY


def test_forward_accepts_x_and_edge_index():
    """Forward accepts (x, edge_index) and returns a 2-tuple."""
    model = GraphCondHeatmap(num_landmarks=9)
    model.eval()

    x = torch.randn(2, 3, 128, 128)
    edge_index = make_chain_edge_index(9)

    with torch.no_grad():
        result = model(x, edge_index)

    assert isinstance(result, tuple)
    assert len(result) == 2
    heatmaps, coords = result
    assert heatmaps.shape[0] == 2
    assert coords.shape[0] == 2


def test_no_num_iters_attribute():
    """Model has no num_iters attribute — single-pass only."""
    model = GraphCondHeatmap(num_landmarks=9)
    assert not hasattr(model, "num_iters")


def test_default_construction_with_only_num_landmarks():
    """Model constructs with only num_landmarks specified."""
    model = GraphCondHeatmap(num_landmarks=5)
    assert model.num_landmarks == 5
    assert model.token_dim == 128  # default gnn_hidden
    assert model.heatmap_size == 64  # default
    assert len(model.gcn_layers) == 2  # default num_layers
    assert model.cross_attn.num_heads == 4  # default num_heads


def test_raises_when_gnn_hidden_not_divisible_by_num_heads():
    """ValueError when gnn_hidden % num_heads != 0."""
    with pytest.raises(ValueError, match="divisible"):
        GraphCondHeatmap(num_landmarks=9, gnn_hidden=100, num_heads=3)


def test_forward_no_initial_coords_required():
    """Forward does NOT require initial_coords parameter."""
    import inspect
    sig = inspect.signature(GraphCondHeatmap.forward)
    params = list(sig.parameters.keys())
    assert "initial_coords" not in params
    assert params == ["self", "x", "edge_index"]


def test_output_coords_in_unit_range():
    """Output coordinates are in [0, 1] range."""
    model = GraphCondHeatmap(num_landmarks=9, gnn_hidden=64, num_heads=4)
    model.eval()

    x = torch.randn(1, 3, 128, 128)
    edge_index = make_chain_edge_index(9)

    with torch.no_grad():
        _, coords = model(x, edge_index)

    assert (coords >= 0.0).all()
    assert (coords <= 1.0).all()


def test_heatmap_shape_matches_config():
    """Heatmap output matches configured heatmap_size."""
    for heatmap_size in [32, 64]:
        model = GraphCondHeatmap(num_landmarks=5, heatmap_size=heatmap_size)
        model.eval()

        x = torch.randn(1, 3, 128, 128)
        edge_index = make_chain_edge_index(5)

        with torch.no_grad():
            heatmaps, _ = model(x, edge_index)

        assert heatmaps.shape == (1, 5, heatmap_size, heatmap_size)
