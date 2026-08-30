"""GraphPriorFusion — Bayesian fusion of appearance heatmaps and graph priors.

Single end-to-end, co-trained model:

    image -> HRNet-w18 backbone -> fused multi-scale map F
      |-- Appearance head (conv heatmap decoder) -> per-landmark likelihood H_i
      |-- Graph-prior head (GCN over anatomical graph) -> per-landmark Gaussian
          prior N(mu_i, Sigma_i) in normalized image coords, with OFFSET-based mean
      -> Bayesian fusion:  P_i(x)  ∝  H_i(x) * N(x; mu_i, Sigma_i)
      -> soft-argmax(P_i) -> coords_i

Key properties (structural, not learned):
- sharp H_i  -> product ≈ H_i  (a broad prior cannot move a confident landmark)
- flat  H_i  -> product ≈ prior (graph rescues an ambiguous landmark)
- bimodal H_i -> prior selects the correct mode

Offset-based prior mean
-----------------------
The graph reasons RELATIVELY: for landmark i it predicts an offset from an
anchor derived from its current belief and its neighbors' beliefs, then anchors
to absolute normalized coords. This keeps the "graph reasons about geometry"
inductive bias while landing the prior in image space so it can multiply H_i.

Sigma-collapse guard
--------------------
The diagonal std is `sigma_min + sigma_span * sigmoid(raw)`, so it is bounded
below by sigma_min (cannot spike to a delta) and above by sigma_min+sigma_span.

NOTE on the ceiling: sigma_span must be large enough that the upper end is
EFFECTIVELY FLAT over the [0,1] coordinate space, otherwise the ceiling does not
mean "prior off" — it mandates a permanently informative prior. With span=0.20
the broadest reachable prior had std 0.21, a blob covering ~44% of the frame at
2-sigma, and the model was pinned there; the continuous "Sigma -> inf reduces to
pure heatmap" limit was unreachable by the optimizer.

But span must ALSO not be so large that useful stds (~0.05-0.25) get squashed into
a saturated tail of the sigmoid. What the optimizer moves is the PRE-ACTIVATION,
and with Adam a parameter travels only ~lr*steps; if the useful band sits many
units away it is unreachable and the prior is frozen at its init. Default span is
0.5: the 0.51 ceiling spans the frame at 2-sigma (genuinely "off"), while useful
stds sit in the well-conditioned middle of the sigmoid. The prior heads also get a
dedicated higher LR (see TrainingEngine._split_prior_param_groups) so they can
actually traverse that band. Collapse is something to DETECT via
last_sigma_trace, not something the parameterization forbids by construction.

A GCN-off mode (prior_disabled=True) reduces the model EXACTLY to the pure
conv-heatmap baseline — this is both the ablation and a correctness check.

Forward signature matches `graph_cond_heatmap`: forward(x, edge_index) ->
(heatmaps, coords), so it reuses the existing heatmap_loss training path.
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import timm

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    raise ImportError(
        "torch_geometric is required for GraphPriorFusion. "
        "Install it with: pip install torch-geometric"
    )

from .registry import register_model
from .hrnet_heatmap import decode_coords, hard_argmax


LANDMARK_EMBED_DIM = 32


@register_model("graph_prior_fusion")
class GraphPriorFusion(nn.Module):
    """Bayesian graph-prior + appearance-heatmap fusion model.

    Args:
        num_landmarks: Number of landmarks (required).
        gnn_hidden: GCN hidden dim / node feature dim.
        num_layers: Number of GCN layers.
        heatmap_size: Output heatmap spatial resolution (appearance likelihood).
        sigma_min: Minimum per-axis prior std in normalized coords (floor).
        sigma_span: Range added to sigma_min (max std = sigma_min + sigma_span).
        offset_scale: Max magnitude of the offset the graph can add to the anchor
            (normalized coords), bounding how far the prior mean can move.
        chol_bias: Initial bias on the Cholesky head. Positive values start the
            prior BROAD (sigmoid(bias) ~ 1 -> std near the sigma_min+sigma_span
            ceiling), which is what makes the start genuinely near-identity:
            a broad prior cannot move a landmark. Zero-init gives sigmoid(0)=0.5,
            i.e. a mid-breadth prior that actively distorts from step 0.
        prior_disabled: If True, skip fusion -> pure conv-heatmap baseline
            (GCN-off ablation / equivalence check).
    """

    def __init__(
        self,
        num_landmarks: int,
        gnn_hidden: int = 128,
        num_layers: int = 2,
        heatmap_size: int = 64,
        sigma_min: float = 0.01,
        sigma_span: float = 0.5,
        offset_scale: float = 0.25,
        chol_bias: float = 2.0,
        decode_mode: str = "windowed",
        decode_radius: int = 5,
        bn_momentum: float = 0.1,
        prior_disabled: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size
        self.token_dim = gnn_hidden
        self.sigma_min = sigma_min
        self.sigma_span = sigma_span
        self.offset_scale = offset_scale
        self.chol_bias = chol_bias
        self.decode_mode = decode_mode
        self.decode_radius = decode_radius
        self.prior_disabled = prior_disabled
        # Runtime toggle used for fusion warm-up (see set_prior_active). Distinct
        # from prior_disabled, which is a permanent architectural ablation.
        self.prior_active = True
        self.last_sigma_trace = torch.tensor(float("nan"))
        self.last_fused_logits = None

        # --- Backbone ---
        self.backbone = timm.create_model(
            "hrnet_w18", pretrained=True, features_only=True,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            feats = self.backbone(dummy)
            H0, W0 = feats[0].shape[2], feats[0].shape[3]
            fused_dummy = torch.cat([
                feats[0],
                F.interpolate(feats[1], size=(H0, W0), mode="bilinear", align_corners=False),
                F.interpolate(feats[2], size=(H0, W0), mode="bilinear", align_corners=False),
                F.interpolate(feats[3], size=(H0, W0), mode="bilinear", align_corners=False),
            ], dim=1)
            self.fused_channels = fused_dummy.shape[1]

        # --- Appearance head (conv heatmap decoder -> likelihood) ---
        # bn_momentum default 0.1 (PyTorch default), NOT the 0.01 used by the
        # paper-faithful WFLW head: this head is trained from scratch on small
        # datasets at small batch size, where momentum 0.01 leaves the running
        # stats unconverged for hundreds of steps and eval() then normalizes with
        # the wrong statistics (measured 15.5x train/eval logit-scale gap).
        self.appearance_head = nn.Sequential(
            nn.Conv2d(self.fused_channels, self.fused_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.fused_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fused_channels, num_landmarks, kernel_size=1),
        )
        nn.init.normal_(self.appearance_head[-1].weight, std=0.001)
        nn.init.constant_(self.appearance_head[-1].bias, 0)

        # --- Graph-prior head ---
        # Node features: per-landmark pooled appearance descriptor + landmark id
        #   + current-belief location (anchor) -> GCN -> (offset, cholesky(3)).
        self.feat_proj = nn.Conv2d(self.fused_channels, gnn_hidden, kernel_size=1)
        self.landmark_embed = nn.Embedding(num_landmarks, LANDMARK_EMBED_DIM)
        node_in = gnn_hidden + LANDMARK_EMBED_DIM + 2  # +2 for anchor (x, y)
        self.node_proj = nn.Linear(node_in, gnn_hidden)
        self.gcn_layers = nn.ModuleList(
            [GCNConv(gnn_hidden, gnn_hidden) for _ in range(num_layers)]
        )
        # Heads: 2 for offset, 3 for Cholesky (log_L11, L21, log_L22).
        self.offset_head = nn.Linear(gnn_hidden, 2)
        self.chol_head = nn.Linear(gnn_hidden, 3)
        # Offset head zero-init -> prior mean starts exactly at the anchor.
        nn.init.zeros_(self.offset_head.weight); nn.init.zeros_(self.offset_head.bias)
        # Cholesky head: zero WEIGHTS but a POSITIVE BIAS so the prior starts at
        # (near) maximum breadth. A broad prior is the identity element of the
        # multiplicative fusion, so appearance genuinely dominates at step 0 and
        # the graph has to earn any narrowing. Zero bias would give sigmoid(0)=0.5
        # -> a mid-breadth blob centred on a meaningless initial anchor.
        nn.init.zeros_(self.chol_head.weight)
        nn.init.constant_(self.chol_head.bias, chol_bias)

    def get_fused_map(self, x: Tensor) -> Tensor:
        feat_maps = self.backbone(x)
        H0, W0 = feat_maps[0].shape[2], feat_maps[0].shape[3]
        return torch.cat([
            feat_maps[0],
            F.interpolate(feat_maps[1], size=(H0, W0), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[2], size=(H0, W0), mode="bilinear", align_corners=False),
            F.interpolate(feat_maps[3], size=(H0, W0), mode="bilinear", align_corners=False),
        ], dim=1)

    def _resize_heatmaps(self, heatmaps: Tensor, H: int, W: int) -> Tensor:
        if self.heatmap_size is not None and H != self.heatmap_size:
            return F.interpolate(
                heatmaps, size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear", align_corners=False,
            )
        return heatmaps

    def _pool_landmark_features(self, feat_map: Tensor, anchors: Tensor) -> Tensor:
        """Bilinear-sample the projected feature map at each landmark anchor.

        Args:
            feat_map: (B, gnn_hidden, H, W) projected features.
            anchors: (B, N, 2) normalized [0,1] anchor locations (x, y).
        Returns:
            (B, N, gnn_hidden) sampled per-landmark descriptors.
        """
        grid = anchors * 2.0 - 1.0  # to [-1, 1]
        grid = grid.unsqueeze(2)  # (B, N, 1, 2)
        sampled = F.grid_sample(feat_map, grid, align_corners=True, mode="bilinear")
        return sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, gnn_hidden)

    def _graph_prior(self, feat_proj_map, anchors, edge_index):
        """Predict per-landmark offset-based Gaussian prior (mu, L) from the graph.

        Returns:
            mu: (B, N, 2) prior mean in normalized coords (anchor + bounded offset).
            L11, L21, L22: (B, N) Cholesky entries of the covariance in normalized coords.
        """
        B, _, _, _ = feat_proj_map.shape
        N = self.num_landmarks

        node_desc = self._pool_landmark_features(feat_proj_map, anchors)  # (B,N,hid)
        lm_ids = torch.arange(N, device=anchors.device)
        lm_emb = self.landmark_embed(lm_ids).unsqueeze(0).expand(B, -1, -1)
        node_feats = torch.cat([node_desc, lm_emb, anchors], dim=-1)
        node_feats = F.relu(self.node_proj(node_feats))

        flat = node_feats.view(B * N, -1)
        batch_edge_index = torch.cat([edge_index + b * N for b in range(B)], dim=1)
        h = flat
        for layer in self.gcn_layers:
            h = F.relu(layer(h, batch_edge_index))
        h = h.view(B, N, -1)

        # Offset-based mean: anchor + bounded offset (tanh * offset_scale).
        offset = torch.tanh(self.offset_head(h)) * self.offset_scale  # (B,N,2)
        mu = torch.clamp(anchors + offset, 0.0, 1.0)

        # Bounded covariance: std in [sigma_min, sigma_min + sigma_span].
        chol = self.chol_head(h)  # (B, N, 3)
        L11 = self.sigma_min + self.sigma_span * torch.sigmoid(chol[..., 0])
        L22 = self.sigma_min + self.sigma_span * torch.sigmoid(chol[..., 2])
        L21 = torch.tanh(chol[..., 1]) * L22  # keep off-diagonal bounded rel. to L22
        return mu, L11, L21, L22

    def _gaussian_prior_map(self, mu, L11, L21, L22, H, W, device):
        """Evaluate the 2D Gaussian prior on the heatmap grid (log-space).

        Sigma = L L^T with L = [[L11,0],[L21,L22]]. We compute the Mahalanobis
        quadratic form per grid cell and return log N (unnormalized) of shape
        (B, N, H, W) so it can be added to heatmap logits before softmax.
        """
        B, N, _ = mu.shape
        xs = torch.linspace(0, 1, W, device=device)
        ys = torch.linspace(0, 1, H, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
        gx = gx.view(1, 1, H, W)
        gy = gy.view(1, 1, H, W)

        dx = gx - mu[..., 0].view(B, N, 1, 1)
        dy = gy - mu[..., 1].view(B, N, 1, 1)

        # Sigma^{-1} from Cholesky: solve L z = d, quad = z^T z.
        L11e = L11.view(B, N, 1, 1) + 1e-8
        L22e = L22.view(B, N, 1, 1) + 1e-8
        L21e = L21.view(B, N, 1, 1)
        z1 = dx / L11e
        z2 = (dy - L21e * z1) / L22e
        quad = z1 * z1 + z2 * z2  # (B, N, H, W)
        return -0.5 * quad  # unnormalized log-Gaussian

    def _decode(self, logits: Tensor) -> Tensor:
        """Decode coordinates from (fused or appearance) logits."""
        return decode_coords(
            logits, mode=self.decode_mode, radius=self.decode_radius
        )

    def set_prior_active(self, active: bool) -> None:
        """Enable/disable the fusion at runtime (used for warm-up).

        During warm-up the model behaves as a pure conv-heatmap model, so the
        appearance head can learn real peaks BEFORE the graph prior starts
        acting on them. This matters because the prior is centred on an anchor
        derived from the appearance heatmap: at init that heatmap is ~uniform,
        its soft-argmax is the image centre, and a prior centred there is a
        self-reinforcing fixed point (the anchor is detached, so nothing pulls
        it off centre directly).
        """
        self.prior_active = bool(active)

    def forward(self, x: Tensor, edge_index: Tensor):
        """Returns (appearance_heatmaps, coords). coords in [0,1].

        IMPORTANT — which tensor goes where:
        The first return value is the APPEARANCE head's raw logits, NOT the fused
        log-posterior. The training path feeds this slot to `heatmap_loss`, which
        MSEs it against a Gaussian target in [0, 1]. Fused logits are log-space
        (log_softmax + log_gaussian, i.e. always <= 0 and typically -10..-30), so
        MSE-ing them against a [0, 1] target is not a valid objective: it is
        unsatisfiable, it dominates the total loss, and its only available descent
        direction is to flatten the prior to its breadth ceiling. That produced a
        frozen train loss and a centre-blob collapse whose radius equalled the
        sigma ceiling.

        Coordinates ARE decoded from the fused posterior, so the graph prior is
        still fully in the gradient path via the coordinate term. This matches the
        design spec: heatmap loss on H_i, coordinate loss on soft-argmax(P_i).

        Also stashes:
          self.last_sigma_trace  — mean trace of Sigma, for collapse monitoring.
          self.last_fused_logits — the fused log-posterior, for debugging or an
                                   optional posterior NLL term.
        """
        fused_map = self.get_fused_map(x)  # (B, C, H, W)
        heatmaps = self.appearance_head(fused_map)  # (B, N, H, W) logits
        heatmaps = self._resize_heatmaps(heatmaps, fused_map.shape[2], fused_map.shape[3])
        B, N, H, W = heatmaps.shape

        # prior_disabled = permanent ablation; prior_active = warm-up toggle.
        if self.prior_disabled or not self.prior_active:
            # GCN-off: pure conv-heatmap baseline (equivalence check / ablation).
            self.last_sigma_trace = torch.tensor(float("inf"))
            self.last_fused_logits = heatmaps
            return heatmaps, self._decode(heatmaps)

        # Anchor = appearance peak (current belief before fusion). Uses ARGMAX, not
        # soft-argmax: the anchor is already detached so it needs no gradient, and
        # soft-argmax over the full map returns the image centre for any map that is
        # not sharply peaked. A centre anchor is a self-reinforcing fixed point --
        # the prior gets centred there, which suppresses appearance evidence
        # elsewhere, which keeps the anchor at the centre. Measured on ideal target
        # maps: argmax anchor ~9px vs soft-argmax anchor ~274px.
        anchors = hard_argmax(heatmaps).detach()  # (B, N, 2)

        feat_proj_map = self.feat_proj(fused_map)  # (B, hid, H, W)
        mu, L11, L21, L22 = self._graph_prior(feat_proj_map, anchors, edge_index)

        # Fuse in log-space: log P = log-softmax(H) + log N(mu, Sigma); then softmax.
        log_prior = self._gaussian_prior_map(mu, L11, L21, L22, H, W, x.device)
        fused_logits = F.log_softmax(heatmaps.view(B, N, -1), dim=-1).view(B, N, H, W) + log_prior

        coords = self._decode(fused_logits)

        # Monitoring: mean trace of Sigma = L11^2 + L21^2 + L22^2.
        self.last_sigma_trace = (L11 ** 2 + L21 ** 2 + L22 ** 2).mean().detach()
        self.last_fused_logits = fused_logits

        return heatmaps, coords
