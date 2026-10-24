"""AttnLocalRefine — two-stage landmark model: frozen global placement + a
high-resolution LOCAL ATTENTION sub-pixel refiner.

Why this exists (and why it is deliberately NOT a GCN)
------------------------------------------------------
Every HRNet+GCN variant in this repo (fused_global, _ms, _patch, hinit, ...)
reasons about landmarks with a static-graph GCN AND samples features that top
out at HRNet branch-0 resolution = input/4. Their own docstrings concede the
same ceiling: multi-scale and multi-offset sampling widen the capture radius but
"do not raise the sub-cell precision ceiling" below input/4. The measured error
sits at ~6-7px across heads regardless of the relational mechanism, which is the
signature of a resolution floor set UNDERNEATH the relational module, not by it.

This model attacks that floor directly and changes the relational mechanism at
the same time, on two axes the GCN variants never separated:

  Stage A (GLOBAL, frozen):  coarse placement from an existing trained model.
     - Resolves configural ambiguity (which digit, overall pose).
     - Frozen so this prototype isolates the ONE new hypothesis below.

  Stage B (LOCAL, trained):  sub-pixel refinement via windowed cross-attention
     over a DEDICATED HIGH-RESOLUTION feature branch at input/1.
     - This is the piece no existing model has: it returns to full-resolution
       pixels that the input/4 fused map physically cannot represent.
     - Relational reasoning is ATTENTION, not GCNConv on a hand-drawn graph.

Easy->hard intuition, without autoregression
---------------------------------------------
Stage A emits a per-landmark confidence (peak sharpness proxy). Stage B's
inter-landmark self-attention is biased by that confidence so information flows
preferentially from confident anchors (e.g. the toepad tip) to uncertain
dependents (mushy proximal base). The easy-first ordering is thus an emergent
attention weighting, computed per image, with no imposed sequence and no
exposure bias.

Forward contract
----------------
    forward(imgs) -> (coarse_coords, final_coords)

Both are (B, N, 2) normalized [0, 1] (x, y). The training engine supervises
final_coords and (optionally, with a small weight) coarse_coords so Stage B
cannot paper over a degenerate Stage A. Image-only forward keeps it off the
GCN dispatch path in the engine.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .registry import register_model, get_model


# ----------------------------------------------------------------------------
# High-resolution feature branch (the floor-breaker)
# ----------------------------------------------------------------------------
class HighResStem(nn.Module):
    """A light conv branch that keeps spatial resolution at input/1.

    Max-performance choice: no downsampling at all, so the window Stage B crops
    is at the native pixel grid. Dilated convs grow the receptive field without
    throwing away resolution. Channel count is modest to keep memory sane when
    the feature map is full-size (512x512 at input_size=512).
    """

    def __init__(self, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(48, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, out_ch, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
        )
        self.out_ch = out_ch

    def forward(self, x):  # (B, 3, H, W) -> (B, out_ch, H, W)
        return self.net(x)


# ----------------------------------------------------------------------------
# Local windowed cross-attention refiner
# ----------------------------------------------------------------------------
class WindowCrossAttention(nn.Module):
    """One multi-head attention block: landmark query attends over the tokens of
    its own high-resolution window.

    Query  : per-landmark token (identity embedding + current-coord encoding).
    Keys/Vs: the K*K high-res feature cells sampled around the current coord,
             each tagged with a relative-position encoding so the block can
             reason about WHERE in the window the evidence is.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, q, kv):
        # q: (BN, 1, dim), kv: (BN, K*K, dim)
        q2, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + q2
        q = q + self.ffn(self.norm_ffn(q))
        return q


@register_model("attn_local_refine")
class AttnLocalRefine(nn.Module):
    """Frozen global placement + high-res local-attention sub-pixel refiner.

    Args:
        num_landmarks: Number of landmarks (required).
        stage_a_variant: Registry name of the frozen global model.
        stage_a_checkpoint: Path to its trained weights (required for real runs;
            may be empty for a smoke test, in which case Stage A is random).
        stage_a_kwargs: Extra constructor kwargs forwarded to Stage A.
        hires_ch: Channel width of the high-res stem (input/1 feature map).
        window: Odd window side length K in high-res pixels.
        dim: Attention working width.
        num_heads: Attention heads.
        num_blocks: Stacked window-attention blocks.
        num_refine_iters: Re-sample-and-refine passes (each re-crops the window
            at the updated coordinate, so a first-pass correction widens the
            basin for the second).
        max_offset_frac: Bound on a single refinement step as a fraction of the
            window half-width, applied via tanh. Keeps Stage B a LOCAL refiner.
    """

    def __init__(
        self,
        num_landmarks: int,
        stage_a_variant: str = "fused_global",
        stage_a_checkpoint: str = "",
        stage_a_kwargs: dict | None = None,
        mean_shape: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        hires_ch: int = 64,
        window: int = 17,
        dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        num_refine_iters: int = 2,
        max_offset_frac: float = 0.5,
        input_size: int = 512,
        **kwargs,
    ):
        super().__init__()
        assert window % 2 == 1, "window must be odd so it has a center cell"
        self.num_landmarks = num_landmarks
        self.window = window
        self.half = window // 2
        self.num_refine_iters = num_refine_iters
        self.max_offset_frac = max_offset_frac
        self.input_size = input_size

        # --- Stage A: frozen global placement -------------------------------
        self.stage_a = get_model(
            stage_a_variant,
            num_landmarks=num_landmarks,
            **(stage_a_kwargs or {}),
        )
        if stage_a_checkpoint:
            state = torch.load(stage_a_checkpoint, map_location="cpu")
            # Checkpoints in this repo are saved as raw state_dicts.
            self.stage_a.load_state_dict(state)
        for p in self.stage_a.parameters():
            p.requires_grad = False
        self.stage_a.eval()
        self._stage_a_needs_graph = stage_a_variant in (
            "fused_global", "fused_global_ms", "fused_global_patch",
            "fused", "multiscale", "standard", "hinit",
        )

        # Mean-shape init for Stage A (buffers so they move with .to(device)).
        if mean_shape is None:
            mean_shape = torch.full((num_landmarks, 2), 0.5)
        self.register_buffer("mean_shape", mean_shape.float(), persistent=False)
        if edge_index is None:
            # Fallback chain topology; the engine overrides this with the real one.
            idx = torch.arange(num_landmarks)
            edge_index = torch.stack(
                [torch.cat([idx[:-1], idx[1:]]), torch.cat([idx[1:], idx[:-1]])]
            )
        self.register_buffer("edge_index", edge_index.long(), persistent=False)

        # --- Stage B: high-res local attention refiner ----------------------
        self.hires = HighResStem(out_ch=hires_ch)

        self.landmark_embed = nn.Embedding(num_landmarks, dim)
        self.coord_embed = nn.Linear(2, dim)
        # Project a sampled high-res feature cell -> attention width.
        self.kv_proj = nn.Linear(hires_ch, dim)
        # Learned relative-position encoding over the K*K window cells.
        self.rel_pos = nn.Parameter(torch.zeros(window * window, dim))
        nn.init.trunc_normal_(self.rel_pos, std=0.02)

        # Inter-landmark self-attention (the relational prior; replaces the GCN).
        self.landmark_sa = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.sa_norm = nn.LayerNorm(dim)

        self.blocks = nn.ModuleList(
            [WindowCrossAttention(dim, num_heads) for _ in range(num_blocks)]
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2)
        )
        # Final offset layer: standard init, zero bias only. Earlier tries at a
        # zero/near-zero final WEIGHT (to start predictions on the Stage-A anchor)
        # throttle d(delta)/d(input) to ~0 and starve the entire Stage-B feature
        # path (stem, kv_proj, attention blocks) of gradient. The refiner stays
        # local not because the init is tiny but because `delta = tanh(raw)*max_norm`
        # HARD-BOUNDS every step to a fraction of the window; that bound is the
        # safety mechanism, so the init can be full-strength for fast learning.
        nn.init.xavier_uniform_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

    # -- Stage A invocation ---------------------------------------------------
    @torch.no_grad()
    def _run_stage_a(self, x):
        B = x.shape[0]
        init = self.mean_shape.unsqueeze(0).expand(B, -1, -1).contiguous()
        if self._stage_a_needs_graph:
            coarse = self.stage_a(x, init, self.edge_index)
            if isinstance(coarse, tuple):
                coarse = coarse[0]
            conf = None
        else:
            out = self.stage_a(x)
            if isinstance(out, tuple):
                # (heatmaps, coords): derive a confidence proxy from peak sharpness
                heatmaps, coarse = out[0], out[1]
                conf = self._heatmap_confidence(heatmaps)
            else:
                coarse, conf = out, None
        return coarse.detach(), (conf.detach() if conf is not None else None)

    @staticmethod
    def _heatmap_confidence(heatmaps):
        """Peak-sharpness proxy in [0,1]: softmax max over each map."""
        B, K, H, W = heatmaps.shape
        p = F.softmax(heatmaps.view(B, K, -1), dim=-1)
        return p.max(dim=-1).values  # (B, K)

    # -- Window sampling ------------------------------------------------------
    def _sample_window(self, feat, coords):
        """Sample a (K x K) high-res window around each landmark.

        Args:
            feat: (B, C, H, W) high-res feature map.
            coords: (B, N, 2) normalized [0,1] (x, y).
        Returns:
            (B, N, K*K, C) sampled window features.
        """
        B, C, H, W = feat.shape
        N = coords.shape[1]
        K = self.window
        device = feat.device

        # Window offsets in pixels, centered, shape (K*K, 2) as (dx, dy).
        rng = torch.arange(-self.half, self.half + 1, device=device, dtype=torch.float32)
        dy, dx = torch.meshgrid(rng, rng, indexing="ij")
        off = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1)  # (K*K, 2)
        # Pixel offset -> normalized [-1,1] grid delta.
        off_norm = torch.stack(
            [off[:, 0] * (2.0 / max(W - 1, 1)), off[:, 1] * (2.0 / max(H - 1, 1))],
            dim=-1,
        )  # (K*K, 2)

        base = coords * 2.0 - 1.0  # (B, N, 2) in [-1,1]
        grid = base.unsqueeze(2) + off_norm.view(1, 1, K * K, 2)  # (B, N, K*K, 2)
        sampled = F.grid_sample(
            feat, grid, align_corners=True, mode="bilinear", padding_mode="border"
        )  # (B, C, N, K*K)
        return sampled.permute(0, 2, 3, 1)  # (B, N, K*K, C)

    # -- Forward --------------------------------------------------------------
    def forward(self, x):
        B = x.shape[0]
        N = self.num_landmarks
        dim = self.landmark_embed.embedding_dim

        coarse, conf = self._run_stage_a(x)
        coords = coarse.clone()

        hires = self.hires(x)  # (B, Chr, H, W) at input/1

        lm_ids = torch.arange(N, device=x.device)
        lm_tok = self.landmark_embed(lm_ids).unsqueeze(0).expand(B, -1, -1)  # (B,N,dim)
        rel = self.rel_pos.unsqueeze(0)  # (1, K*K, dim)

        # Confidence bias for the inter-landmark self-attention: let low-confidence
        # landmarks be weak KEYS so information flows from anchors to dependents.
        if conf is not None:
            # (B, N) -> additive bias broadcast over query dim, in log space.
            key_bias = torch.log(conf.clamp_min(1e-4)).unsqueeze(1)  # (B,1,N)
        else:
            key_bias = None

        for _ in range(self.num_refine_iters):
            win = self._sample_window(hires, coords)  # (B,N,K*K,Chr)
            kv = self.kv_proj(win) + rel  # (B,N,K*K,dim)

            q = (lm_tok + self.coord_embed(coords)).unsqueeze(2)  # (B,N,1,dim)

            # Flatten landmarks into the batch for per-landmark window attention.
            qf = q.reshape(B * N, 1, dim)
            kvf = kv.reshape(B * N, self.window * self.window, dim)
            for blk in self.blocks:
                qf = blk(qf, kvf)
            tok = qf.reshape(B, N, dim)

            # Inter-landmark relational pass (attention, NOT a GCN).
            attn_mask = None
            tok_n = self.sa_norm(tok)
            if key_bias is not None:
                # Build an additive (B*heads, N, N) mask is overkill for a proto;
                # fold confidence into values by scaling keys instead.
                tok_kv = tok_n * conf.clamp_min(1e-4).unsqueeze(-1)
            else:
                tok_kv = tok_n
            sa, _ = self.landmark_sa(tok_n, tok_kv, tok_kv, attn_mask=attn_mask)
            tok = tok + sa

            # Bounded sub-pixel offset in normalized units.
            raw = self.offset_head(tok)  # (B,N,2)
            # Half-window extent in normalized coords (x and y share H=W here).
            max_norm = self.max_offset_frac * self.half * (2.0 / max(self.input_size - 1, 1)) / 2.0
            delta = torch.tanh(raw) * max_norm
            coords = (coords + delta).clamp(0.0, 1.0)

        return coarse, coords

    def train(self, mode: bool = True):
        """Keep the frozen Stage A in eval mode regardless of parent mode."""
        super().train(mode)
        self.stage_a.eval()
        return self
