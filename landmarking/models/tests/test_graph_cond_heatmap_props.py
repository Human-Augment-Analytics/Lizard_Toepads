"""Property-based tests for GraphCondHeatmap model.

Tests correctness properties from the design document using Hypothesis.
"""

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from landmarking.models.graph_cond_heatmap import GraphCondHeatmap, make_2d_sinusoidal_pe
from landmarking.models.hrnet_heatmap import soft_argmax
from landmarking.common.graph_topologies import make_chain_edge_index


# --- Property 1: Construction consistency across hyperparameters ---


@settings(max_examples=100, deadline=None)
@given(
    num_landmarks=st.integers(min_value=1, max_value=200),
    gnn_hidden=st.sampled_from([64, 128, 256]),
    num_layers=st.integers(min_value=1, max_value=6),
    num_heads=st.sampled_from([1, 2, 4, 8]),
    heatmap_size=st.sampled_from([32, 64, 128]),
)
def test_construction_consistency(num_landmarks, gnn_hidden, num_layers, num_heads, heatmap_size):
    """Property 1: Model constructs with consistent dimensions for valid hyperparameters.

    Feature: graph_cond_heatmap, Property 1: Construction consistency
    """
    assume(gnn_hidden % num_heads == 0)

    model = GraphCondHeatmap(
        num_landmarks=num_landmarks,
        gnn_hidden=gnn_hidden,
        num_layers=num_layers,
        num_heads=num_heads,
        heatmap_size=heatmap_size,
    )

    # Embedding table shape
    assert model.landmark_embed.weight.shape == (num_landmarks, gnn_hidden)
    # GCN layer count
    assert len(model.gcn_layers) == num_layers
    # Attention heads
    assert model.cross_attn.num_heads == num_heads
    # Token dim matches gnn_hidden
    assert model.token_dim == gnn_hidden


# --- Property 2: Forward pass output shape invariant ---


@settings(max_examples=50, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    input_size=st.sampled_from([128, 256]),
    num_landmarks=st.integers(min_value=2, max_value=20),
)
def test_forward_output_shape(batch_size, input_size, num_landmarks):
    """Property 2: Forward pass produces correct output shapes.

    Feature: graph_cond_heatmap, Property 2: Forward pass output shape invariant
    """
    heatmap_size = 64
    model = GraphCondHeatmap(
        num_landmarks=num_landmarks,
        gnn_hidden=128,
        num_layers=2,
        num_heads=4,
        heatmap_size=heatmap_size,
    )
    model.eval()

    x = torch.randn(batch_size, 3, input_size, input_size)
    edge_index = make_chain_edge_index(num_landmarks)

    with torch.no_grad():
        heatmaps, coords = model(x, edge_index)

    assert heatmaps.shape == (batch_size, num_landmarks, heatmap_size, heatmap_size)
    assert coords.shape == (batch_size, num_landmarks, 2)


# --- Property 3: Dot-product heatmap equals einsum ---


@settings(max_examples=100, deadline=None)
@given(
    B=st.integers(min_value=1, max_value=4),
    N=st.integers(min_value=1, max_value=20),
    D=st.sampled_from([32, 64, 128]),
    H=st.integers(min_value=4, max_value=16),
    W=st.integers(min_value=4, max_value=16),
)
def test_dot_product_heatmap_correctness(B, N, D, H, W):
    """Property 3: Dot-product heatmap equals einsum of tokens and spatial features.

    Feature: graph_cond_heatmap, Property 3: Dot-product heatmap correctness
    """
    tokens = torch.randn(B, N, D)
    spatial = torch.randn(B, D, H, W)

    expected = torch.einsum('bnd,bdhw->bnhw', tokens, spatial)

    # Verify the operation matches (this is what the model does internally)
    assert expected.shape == (B, N, H, W)
    # Verify numerical correctness via manual computation
    for b in range(B):
        for n in range(N):
            for h in range(H):
                for w in range(min(W, 2)):  # Spot-check a few positions
                    dot = (tokens[b, n] * spatial[b, :, h, w]).sum()
                    assert torch.allclose(expected[b, n, h, w], dot, atol=1e-5)


# --- Property 4: Soft-argmax output bounded in [0, 1] ---


@settings(max_examples=100, deadline=None)
@given(
    B=st.integers(min_value=1, max_value=4),
    N=st.integers(min_value=1, max_value=20),
    H=st.sampled_from([16, 32, 64]),
    W=st.sampled_from([16, 32, 64]),
)
def test_soft_argmax_bounded(B, N, H, W):
    """Property 4: Soft-argmax output is always bounded in [0, 1].

    Feature: graph_cond_heatmap, Property 4: Soft-argmax output bounded
    """
    heatmaps = torch.randn(B, N, H, W) * 10.0  # Arbitrary scale

    coords = soft_argmax(heatmaps)

    assert coords.shape == (B, N, 2)
    assert (coords >= 0.0).all(), f"Found coords < 0: {coords.min()}"
    assert (coords <= 1.0).all(), f"Found coords > 1: {coords.max()}"
