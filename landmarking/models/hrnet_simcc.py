"""SimCC-style 1-D coordinate classification on an HRNet backbone.

Why this variant exists
-----------------------
``docs/graph_prior_theory.md`` retires the coordinate-regression
feature-enrichment variants (``fused_global_ms``, ``fused_global_star``,
``fused_global_patch``) as documented negatives and concludes that the remaining
explanations for the error floor are out-of-basin initialization or a
coordinate-representation limit. The established Lizard heatmap baseline sits at
~6.5 px against a measured oracle shape bound of 2.1-2.6 px
(``landmarking/common/shape_space.py``), so there is real headroom and the
representation is a credible place to look for it.

The concrete resolution budget, measured rather than assumed. timm's
``hrnet_w18`` with ``features_only`` emits 5 maps at reductions
[2, 4, 8, 16, 32]; the HRNetV2 fusion used here and in ``HRNetHeatmap``
concatenates the first four at branch-0 resolution, which is input/**2** -- a
256x256 map for the 512px Lizard canvas. The Lizard heatmap configs then set
``heatmap_size: 128``, which *downsamples* that map by 2x before decoding, to 4
canvas px per cell. So there are two separate things capping localization: the
2D grid the coordinates are read off, and the fact that the configs voluntarily
decode below what the backbone produces.

A 2D heatmap pays O(bins^2) for grid resolution, which is why nobody raises it.
SimCC (Li et al., "SimCC: a Simple Coordinate Classification Perspective for
Human Pose Estimation", ECCV 2022) replaces the 2D map with two 1-D
classifications, one per axis, so resolution costs O(bins). At 9 landmarks that
makes fine bins essentially free: ``heatmap_size`` 128 x ``simcc_upsample`` 4 is
512 bins per axis over a 512px canvas -- one bin per canvas pixel -- for
9 x 2 x 512 = 9216 output values.

How this differs from reference SimCC
-------------------------------------
Reference SimCC flattens each keypoint's (H, W) map and applies a ``Linear`` to
the bin axis. At our resolution that is H*W = 16384 inputs -> 512 bins = 8.4M
parameters per axis per layer, and it discards the spatial arrangement entirely
(any permutation of the input cells is an equivalent parameterization, so the
layer has to re-learn that neighbouring cells mean neighbouring positions).

Instead this head MARGINALIZES. The conv head emits a 2D logit map exactly as
``HRNetHeatmap`` does, and the per-axis logits are obtained by log-sum-exp over
the other axis:

    logits_x(w) = log sum_h exp(map(h, w))

If the 2D map is read as an unnormalized log-density over location -- which is
precisely how ``heatmap_loss(mode="ce")`` and ``graph_prior_fusion`` already
treat it -- then log-sum-exp is the exact marginalization operator, not an
approximation. A depthwise ``ConvTranspose1d`` then upsamples each marginal from
``heatmap_size`` to ``num_bins``, preserving locality and costing
``num_landmarks * 2 * simcc_upsample`` weights (72 at the defaults).

Why the decoder is windowed
---------------------------
``docs/graph_prior_theory.md`` lines 756-790 document that a global soft-argmax
over 16384 cells is a centre-predictor at realistic logit scales (265 px error
on PERFECT Gaussian targets). The 1-D case has the same pathology for the same
reason -- a near-uniform softmax over 512 bins has expectation 0.5 -- so the
default readout here is the 1-D analogue of ``windowed_soft_argmax``: argmax
locates the bin (index detached), and the expectation is taken only over
+/-``simcc_decode_radius`` bins around it. ``global`` and ``hard`` modes are kept
for ablation, matching ``decode_coords``.

Bin-centre convention
---------------------
Bin ``i`` is the normalized coordinate ``(i + 0.5) / num_bins``, so a coordinate
``c`` sits at bin ``c * num_bins - 0.5``. ``simcc_loss`` generates its Gaussian
targets with the same convention; changing one without the other introduces a
half-bin systematic bias.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model

# timm's hrnet_w18 features_only emits 5 maps (reductions 2/4/8/16/32); the
# HRNetV2 fusion in this package uses the first four. Kept as a named constant
# because getting it wrong is a shape error at head-construction time.
_NUM_FUSED_BRANCHES = 4


def hard_argmax_1d(logits: torch.Tensor) -> torch.Tensor:
    """Bin-centre of the argmax bin. Not differentiable.

    Args:
        logits: (B, K, L) raw per-bin logits.

    Returns:
        (B, K) coordinates in [0, 1].
    """
    L = logits.shape[-1]
    idx = logits.argmax(dim=-1).float()
    return (idx + 0.5) / L


def soft_argmax_1d(logits: torch.Tensor) -> torch.Tensor:
    """Full-axis softmax expectation.

    WARNING -- centre-biased for the same reason the 2D global soft-argmax is
    (see ``hrnet_heatmap.soft_argmax``): a near-uniform softmax over ``L`` bins
    has expectation 0.5, so this returns the image centre for any axis
    distribution that is not already sharply peaked. Kept for ablation only.

    Args:
        logits: (B, K, L) raw per-bin logits.

    Returns:
        (B, K) coordinates in [0, 1].
    """
    L = logits.shape[-1]
    pos = torch.arange(L, device=logits.device, dtype=logits.dtype)
    centres = (pos + 0.5) / L
    return (F.softmax(logits, dim=-1) * centres.view(1, 1, L)).sum(dim=-1)


def windowed_soft_argmax_1d(logits: torch.Tensor, radius: int = 8) -> torch.Tensor:
    """Softmax expectation restricted to a window around the peak bin.

    The 1-D analogue of ``hrnet_heatmap.windowed_soft_argmax``. The argmax index
    is DETACHED, so no gradient flows through the discrete bin choice, but the
    readout stays differentiable with respect to the logit VALUES inside the
    window. This is what keeps the estimate unbiased: the thousands of
    irrelevant bins that drag a global expectation toward 0.5 are masked out.

    The peak bin is always inside its own window, so no row is fully masked and
    the softmax is well defined.

    Args:
        logits: (B, K, L) raw per-bin logits.
        radius: Window half-width in bins.

    Returns:
        (B, K) coordinates in [0, 1].
    """
    B, K, L = logits.shape
    idx = logits.argmax(dim=-1).detach().float().view(B, K, 1)
    pos = torch.arange(L, device=logits.device, dtype=logits.dtype).view(1, 1, L)

    inside = (pos - idx).abs() <= radius
    masked = logits.masked_fill(~inside, float("-inf"))
    weights = F.softmax(masked, dim=-1)

    centres = (pos + 0.5) / L
    return (weights * centres).sum(dim=-1)


def decode_simcc(
    logits_x: torch.Tensor,
    logits_y: torch.Tensor,
    mode: str = "windowed",
    radius: int = 8,
) -> torch.Tensor:
    """Decode (x, y) from the two per-axis logit tensors.

    Args:
        logits_x: (B, K, Lx) x-axis logits.
        logits_y: (B, K, Ly) y-axis logits.
        mode: "windowed" (default, differentiable and unbiased), "global"
            (centre-biased, ablation only), or "hard" (not differentiable).
        radius: Window half-width in bins for "windowed".

    Returns:
        (B, K, 2) coordinates in [0, 1].

    Raises:
        ValueError: On an unknown mode.
    """
    if mode == "windowed":
        x = windowed_soft_argmax_1d(logits_x, radius=radius)
        y = windowed_soft_argmax_1d(logits_y, radius=radius)
    elif mode == "global":
        x = soft_argmax_1d(logits_x)
        y = soft_argmax_1d(logits_y)
    elif mode == "hard":
        x = hard_argmax_1d(logits_x)
        y = hard_argmax_1d(logits_y)
    else:
        raise ValueError(
            f"Unknown simcc decode mode {mode!r}; expected 'windowed', "
            f"'global', or 'hard'."
        )
    return torch.stack([x, y], dim=-1)


class _AxisUpsample(nn.Module):
    """Depthwise 1-D upsample from feature-grid resolution to bin resolution.

    ``ConvTranspose1d`` with ``kernel_size = 2 * up``, ``stride = up``,
    ``padding = up // 2`` maps length ``L`` to exactly ``L * up``:

        (L - 1) * up - 2 * (up // 2) + 2 * up  ==  L * up      (for even up)

    ``groups = num_landmarks`` keeps each landmark's marginal independent, so a
    landmark cannot be upsampled using another landmark's evidence.

    Initialization matters here. With the usual small-random init the head emits
    near-zero logits, the axis distribution starts uniform, and the windowed
    decoder then centres its window on an arbitrary argmax -- a needlessly bad
    starting point. Initializing every transposed-conv weight to ``0.5`` instead
    makes the layer a linear-interpolation-style smooth upsample at step 0, so
    the head begins as "marginalize and replicate" and training starts from the
    backbone's actual spatial evidence. The refinement conv is applied
    RESIDUALLY with a small-random init, so it starts as a no-op and has to earn
    any deviation.
    """

    def __init__(self, num_landmarks: int, up: int, refine_kernel: int = 5):
        super().__init__()
        self.up = up
        self.deconv = nn.ConvTranspose1d(
            num_landmarks,
            num_landmarks,
            kernel_size=2 * up,
            stride=up,
            padding=up // 2,
            groups=num_landmarks,
            bias=False,
        )
        nn.init.constant_(self.deconv.weight, 0.5)

        self.refine = nn.Conv1d(
            num_landmarks,
            num_landmarks,
            kernel_size=refine_kernel,
            padding=refine_kernel // 2,
            groups=num_landmarks,
        )
        nn.init.normal_(self.refine.weight, std=0.001)
        nn.init.constant_(self.refine.bias, 0.0)

    def forward(self, marginal: torch.Tensor) -> torch.Tensor:
        """(B, K, L) -> (B, K, L * up)."""
        up = self.deconv(marginal)
        return up + self.refine(up)


@register_model("simcc")
class HRNetSimCC(nn.Module):
    """HRNet backbone with a marginalized SimCC (1-D classification) head.

    Forward contract: ``forward(imgs) -> (logits_x, logits_y, coords)`` where the
    logit tensors are ``(B, K, num_bins)`` raw per-bin logits and ``coords`` is
    ``(B, K, 2)`` in [0, 1]. This is a THREE-element tuple, deliberately
    distinct from the heatmap family's ``(heatmaps, coords)``, so a dispatch site
    that forgets to handle SimCC fails loudly on unpacking instead of silently
    treating ``logits_y`` as coordinates.

    Args:
        num_landmarks: Number of landmarks (K).
        pretrained: Load ImageNet backbone weights.
        heatmap_size: Spatial size of the intermediate 2D logit map (H = W). The
            backbone's branch-0 output is interpolated to this if they differ.
            At input 512 branch-0 is 256, so 128 downsamples and 256 is native;
            both are useful arms (see config/experiments/simcc/).
        simcc_upsample: Bin-resolution multiplier. ``num_bins = heatmap_size *
            simcc_upsample``. Must be even and >= 2 for the length arithmetic in
            ``_AxisUpsample`` to land exactly.
        simcc_decode_mode: "windowed" (default), "global", or "hard".
        simcc_decode_radius: Window half-width in BINS for the windowed decoder.
            Note this is in bins, not feature cells: at the defaults one bin is
            one canvas pixel, so 8 means +/-8 px.
        bn_momentum: BatchNorm momentum for the from-scratch head. 0.1 (not the
            paper-faithful HRNet 0.01) because on small batches a momentum-0.01
            BN leaves its running stats unconverged and ``eval()`` then
            misnormalizes -- measured at 15.5x logit-scale discrepancy in
            ``docs/graph_prior_theory.md``.

    Raises:
        ValueError: If ``simcc_upsample`` is odd or < 2.
    """

    def __init__(
        self,
        num_landmarks: int,
        pretrained: bool = True,
        heatmap_size: int = 128,
        simcc_upsample: int = 4,
        simcc_decode_mode: str = "windowed",
        simcc_decode_radius: int = 8,
        bn_momentum: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        if simcc_upsample < 2 or simcc_upsample % 2 != 0:
            raise ValueError(
                f"simcc_upsample must be an even integer >= 2, got "
                f"{simcc_upsample}. The ConvTranspose1d length arithmetic "
                f"(padding = up // 2) only lands on exactly L * up for even up."
            )

        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size
        self.simcc_upsample = simcc_upsample
        self.num_bins = heatmap_size * simcc_upsample
        self.decode_mode = simcc_decode_mode
        self.decode_radius = simcc_decode_radius

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=pretrained,
            features_only=True,
        )

        # Fused channel count from a dummy forward, matching HRNetHeatmap.
        # NOTE timm's hrnet_w18 features_only returns FIVE maps, at reductions
        # [2, 4, 8, 16, 32] with channels [64, 128, 256, 512, 1024]. ``_fuse``
        # concatenates only the first four (960 channels), exactly as
        # HRNetHeatmap does, so this probe must slice to [:4]; summing all five
        # gives 1984 and builds a head that cannot accept its own input.
        with torch.no_grad():
            feats = self.backbone(torch.zeros(1, 3, 64, 64))
            all_channels = sum(f.shape[1] for f in feats[:_NUM_FUSED_BRANCHES])

        # 2D logit map head -- identical in form to HRNetHeatmap.head, so the
        # intermediate representation stays comparable between the two variants.
        self.head = nn.Sequential(
            nn.Conv2d(all_channels, all_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(all_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(all_channels, num_landmarks, kernel_size=1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.001)
        nn.init.constant_(self.head[-1].bias, 0)

        self.up_x = _AxisUpsample(num_landmarks, simcc_upsample)
        self.up_y = _AxisUpsample(num_landmarks, simcc_upsample)

    def _fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone + 4-branch bilinear fusion at branch-0 resolution."""
        feat_maps = self.backbone(x)
        H, W = feat_maps[0].shape[2], feat_maps[0].shape[3]
        return torch.cat([
            feat_maps[0],
            F.interpolate(feat_maps[1], size=(H, W), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[2], size=(H, W), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[3], size=(H, W), mode="bilinear", align_corners=False),
        ], dim=1)

    def forward_logits(self, x: torch.Tensor):
        """Return ``(logits_x, logits_y, map_2d)`` without decoding.

        ``map_2d`` is the intermediate 2D logit map, exposed for visualization
        and for an optional auxiliary 2D term; the training loss does not use it.
        """
        fused = self._fuse(x)
        map_2d = self.head(fused)

        if map_2d.shape[-1] != self.heatmap_size or map_2d.shape[-2] != self.heatmap_size:
            map_2d = F.interpolate(
                map_2d,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )

        # Exact marginalization of an unnormalized log-density. The constant
        # log(H) offset this introduces is irrelevant: the per-axis softmax that
        # consumes these logits is shift-invariant.
        marg_x = torch.logsumexp(map_2d, dim=2)  # over rows -> (B, K, W)
        marg_y = torch.logsumexp(map_2d, dim=3)  # over cols -> (B, K, H)

        return self.up_x(marg_x), self.up_y(marg_y), map_2d

    def forward(self, x: torch.Tensor):
        """(B, 3, S, S) -> ``(logits_x, logits_y, coords)``."""
        logits_x, logits_y, _ = self.forward_logits(x)
        coords = decode_simcc(
            logits_x, logits_y,
            mode=self.decode_mode,
            radius=self.decode_radius,
        )
        return logits_x, logits_y, coords

    def predict_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: forward and return only ``(B, K, 2)`` coordinates."""
        return self.forward(x)[2]
