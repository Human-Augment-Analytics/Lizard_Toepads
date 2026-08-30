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
The covariance is parameterized so it CANNOT grow without bound: the diagonal
std is `sigma_min + sigma_span * sigmoid(raw)`, bounding the prior's breadth. A
GCN-off mode (prior_disabled=True or forcing Sigma huge) reduces the model
EXACTLY to the pure conv-heatmap baseline — this is both the ablation and a
correctness check. The forward also returns the mean predicted Sigma trace for
monitoring.

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
from .hrnet_heatmap import soft_argmax


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
        sigma_span: float = 0.20,
        offset_scale: float = 0.10,
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
        self.prior_disabled = prior_disabled

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
        self.appearance_head = nn.Sequential(
            nn.Conv2d(self.fused_channels, self.fused_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.fused_channels, momentum=0.01),
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
        # Init offset/chol to ~0 so the prior starts broad and near the anchor
        # (near-identity fusion at init -> stable start, appearance dominates first).
        nn.init.zeros_(self.offset_head.weight); nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.chol_head.weight); nn.init.zeros_(self.chol_head.bias)

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

    def forward(self, x: Tensor, edge_index: Tensor):
        """Returns (fused_heatmaps, coords). coords in [0,1].

        Also stashes self.last_sigma_trace (float tensor) for monitoring.
        """
        fused_map = self.get_fused_map(x)  # (B, C, H, W)
        heatmaps = self.appearance_head(fused_map)  # (B, N, H, W) logits
        heatmaps = self._resize_heatmaps(heatmaps, fused_map.shape[2], fused_map.shape[3])
        B, N, H, W = heatmaps.shape

        if self.prior_disabled:
            # GCN-off: pure conv-heatmap baseline (equivalence check / ablation).
            self.last_sigma_trace = torch.tensor(float("inf"))
            return heatmaps, soft_argmax(heatmaps)

        # Anchor = appearance-only soft-argmax (current belief before fusion).
        anchors = soft_argmax(heatmaps).detach()  # (B, N, 2); detach: anchor is a
        # placement, gradients to appearance flow via the heatmap term, not the anchor.

        feat_proj_map = self.feat_proj(fused_map)  # (B, hid, H, W)
        mu, L11, L21, L22 = self._graph_prior(feat_proj_map, anchors, edge_index)

        # Fuse in log-space: log P = log-softmax(H) + log N(mu, Sigma); then softmax.
        log_prior = self._gaussian_prior_map(mu, L11, L21, L22, H, W, x.device)
        fused_logits = F.log_softmax(heatmaps.view(B, N, -1), dim=-1).view(B, N, H, W) + log_prior

        coords = soft_argmax(fused_logits)

        # Monitoring: mean trace of Sigma = L11^2 + L21^2 + L22^2.
        self.last_sigma_trace = (L11 ** 2 + L21 ** 2 + L22 ** 2).mean().detach()

        # Return fused logits as the "heatmaps" so heatmap_loss supervises the fused
        # distribution (and, via log_softmax(H) inside, the appearance head too).
        return fused_logits, coords
