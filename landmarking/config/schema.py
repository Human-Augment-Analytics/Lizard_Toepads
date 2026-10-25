"""Configuration schema for the landmarking package.

Defines nested dataclasses for paths, dataset, model, and training configuration.
Supports JSON serialization and environment variable overrides.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json

from .resolver import resolve_path, resolve_dataset_dir


@dataclass
class PathConfig:
    """Filesystem path configuration."""

    data_root: str = ""
    output_root: str = "./runs"
    yolo_obb_model: str = ""


@dataclass
class DatasetConfig:
    """Dataset-specific configuration."""

    name: str = "lizard"
    num_landmarks: int = 9
    input_size: int = 512
    graph_topology: str = "chain"
    mean_shape_path: str = ""
    flip_pairs: list = field(default_factory=list)
    data_dir: str = ""
    split_path: str = ""
    landmark_indices: list = field(default_factory=list)
    pixel_spacing: float = 0.1


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    variant: str = "fused"
    feat_dim: int = 64
    gnn_hidden: int = 128
    num_layers: int = 3
    num_iters: int = 4
    scale_indices: list = field(default_factory=lambda: [0, 1, 2, 3])
    use_coarse_init: bool = True
    heatmap_checkpoint: str = ""
    backbone_pretrained_path: str = ""
    heatmap_size: int = 64
    num_heads: int = 4
    sigma: float = 1.5
    patch_mode: str = "multiradius"
    patch_step: int = 1
    patch_radius: int = 2
    patch_radii: list = field(default_factory=lambda: [2, 6, 14])
    patch_proj_dim: int = 32
    prior_sigma_min: float = 0.01
    # Prior std = sigma_min + sigma_span * sigmoid(raw), so the reachable band is
    # [sigma_min, sigma_min + sigma_span]. Two competing requirements:
    #   - the top of the band must be broad enough to be effectively FLAT over the
    #     [0,1] coordinate space (so "broad" really means "prior off"). span=0.20
    #     failed this: its 0.21 ceiling is a blob covering ~44% of the frame.
    #   - the band must not be so wide that useful stds (~0.05-0.25) are squashed
    #     into a saturated corner of the sigmoid the optimizer cannot travel to.
    # span=0.5 satisfies both: ceiling 0.51 spans the frame at 2-sigma, while
    # useful stds sit in the well-conditioned middle of the sigmoid.
    prior_sigma_span: float = 0.5
    # Max distance (normalized coords) the graph may move the prior mean away from
    # the appearance anchor. This bound exists to limit damage from a bad anchor;
    # now that anchors come from argmax rather than centre-biased soft-argmax it can
    # be looser, which matters because rescuing a wrong appearance peak REQUIRES a
    # large offset. At 0.10 the prior mean was confined to +/-102px at canvas 1024.
    prior_offset_scale: float = 0.25
    # Positive bias on the Cholesky head -> prior starts broad (near-identity
    # fusion), so appearance dominates first and the graph must earn any narrowing.
    # Paired with lr_prior_mult so it can actually travel to an informative std.
    prior_chol_bias: float = 2.0
    # Number of initial epochs with fusion disabled, letting the appearance head
    # learn real peaks before the graph prior acts on its anchors.
    prior_warmup_epochs: int = 0
    prior_disabled: bool = False
    # Coordinate decoder for heatmap-family models. "windowed" = soft-argmax within
    # decode_radius cells of the argmax peak (differentiable AND unbiased);
    # "global" = legacy full-map soft-argmax (severely centre-biased, kept for
    # ablations); "hard" = argmax with sub-pixel refinement (not differentiable).
    decode_mode: str = "windowed"
    decode_radius: int = 5
    # BatchNorm momentum for from-scratch heatmap heads. 0.01 is the paper-faithful
    # HRNet value and is fine on large datasets; on small datasets with small
    # batches it leaves running stats unconverged, so eval() misnormalizes.
    bn_momentum: float = 0.1
    # --- PIPNet (variant "pipnet") only ---------------------------------------
    # torchvision backbone architecture (resnet18/34/50/101). PIPNet uses
    # torchvision, not timm, and unpacks the ResNet layers manually.
    backbone: str = "resnet18"
    # Grid stride: grid = input_size / net_stride. 32 -> no extra layers; 64/128
    # append strided convs; 16 appends a transposed conv (reference net_stride).
    net_stride: int = 32
    # NRM neighbors per landmark. Reference default is 10 (for ~98 landmarks); the
    # Lizard config overrides this to 8 since N=9 (a landmark cannot have more
    # than N-1 distinct neighbors).
    num_nb: int = 10
    # PIPNet + STAR (Option A): when True, add an anisotropic covariance head and
    # a STAR term on the DECODED coordinates, on top of the unchanged PIPNet
    # score/offset losses. False keeps the paper-faithful PIPNet exactly.
    pipnet_use_star: bool = False
    # HRNet heatmap + STAR (Option A): when True, add an anisotropic covariance
    # head and a STAR term on the decoded coords, on top of the unchanged heatmap
    # CE/coord loss. False keeps the paper-faithful heatmap variant exactly.
    heatmap_use_star: bool = False
    # --- HRNet cascade (variant "hrnet_cascade") only -------------------------
    # Cascaded heatmap refinement on top of HRNet features. num_stages = K
    # refinement stages; shared_weights = recurrent (one shared stage) vs
    # independent per-stage params (shared is the safer small-data default);
    # cascade_width = the refinement feature-stream width. Reuses heatmap_size,
    # decode_mode, decode_radius, and bn_momentum above.
    num_stages: int = 3
    shared_weights: bool = True
    cascade_width: int = 256
    # --- SimCC (variant "simcc") only -----------------------------------------
    # 1-D coordinate classification. num_bins = heatmap_size * simcc_upsample per
    # axis, so resolution costs O(bins) rather than the O(bins^2) a 2D heatmap
    # pays. Must be EVEN and >= 2 (the depthwise ConvTranspose1d length
    # arithmetic only lands exactly for even factors). At heatmap_size 128 and
    # upsample 4 that is 512 bins over the 512px Lizard canvas -- one bin per
    # canvas pixel, which removes the input/4 quantization ceiling that every
    # other heatmap-family variant here is pinned to.
    simcc_upsample: int = 4
    # Readout over the bin axis. "windowed" = softmax expectation within
    # simcc_decode_radius bins of the peak (differentiable AND unbiased).
    # "global" has exactly the centre-bias pathology documented for the 2D
    # soft-argmax (a near-uniform softmax over 512 bins has expectation 0.5), so
    # it is for ablation only. "hard" = argmax bin centre, not differentiable.
    simcc_decode_mode: str = "windowed"
    # Window half-width in BINS, not feature cells. At the Lizard defaults one
    # bin is one canvas pixel, so 8 means +/-8px.
    simcc_decode_radius: int = 8
    # Gaussian target width in BINS. Does NOT carry over from model.sigma, which
    # is in heatmap cells: sigma 5.0 at heatmap_size 128 is 20 canvas px, and
    # reproducing that breadth here (sigma 20) would discard most of the
    # resolution this variant exists to buy. 6.0 = a 6px target at the defaults.
    simcc_sigma: float = 6.0
    # --- Shape projection (variant "heatmap_shape") only -----------------------
    # Constrains decoded coordinates to a point-distribution model fitted offline
    # by landmarking.scripts.shape_bound_report --save-basis. Required: there is
    # no sensible default basis, and a wrong one silently changes the geometry.
    shape_basis_path: str = ""
    # Group inside the basis file. "all" pools finger and toe; single-class
    # bases measure better (a pooled basis spends 79% of its leading component
    # on the finger/toe distinction) but need one model per class.
    shape_basis_group: str = "all"
    # "similarity" = basis fitted with rotation factored out (strongly preferred:
    # residual rotation after the OBB crop is the dominant variance direction, so
    # letting the fitted transform absorb it keeps the coefficients for shape).
    # "no_rotation" = rotation counted as shape variation.
    shape_basis_variant: str = "similarity"
    # Components retained (k). The measured oracle bound on Lizard: k=6 gives
    # 2.6 px (finger) / 2.1 px (toe), k=9 gives 1.4 / 1.2, against ~6.5 px for
    # the unconstrained heatmap model. Budget headroom above the bound.
    shape_components: int = 8
    # Active Shape Model plausibility clamp, in standard deviations. 0 disables.
    # A saturated coefficient gets zero gradient, so overly tight values stall
    # learning.
    shape_n_sigma: float = 3.0
    # 1.0 = hard projection (the regime the oracle bound describes); lower makes
    # the prior a soft regularizer blended with the raw prediction.
    shape_blend: float = 1.0
    # Alternating refinement steps inside the projection. Cheap relative to the
    # backbone (20 steps ~= 14% of one head conv), so default generously.
    shape_fit_iters: int = 20
    # --- AttnLocalRefine (variant "attn_local_refine") only --------------------
    # Two-stage: a FROZEN global model places coarse coords, then a high-res
    # local-attention head predicts sub-pixel offsets. Stage A is any registered
    # coordinate/heatmap model; its checkpoint and the hyperparameters it was
    # trained with are specified here so the loaded architecture matches.
    stage_a_variant: str = "fused_global"
    stage_a_checkpoint: str = ""
    stage_a_feat_dim: int = 64
    stage_a_gnn_hidden: int = 256
    stage_a_num_layers: int = 3
    stage_a_num_iters: int = 4
    # Stage B (the floor-breaker). hires_ch = channels of the input/1 conv stem;
    # window = odd side length (in high-res px) of the per-landmark attention
    # window; attn_dim/num_blocks/num_refine_iters size the refiner; max_offset_frac
    # bounds a single step to this fraction of the window half-width (keeps it a
    # LOCAL refiner, not a second global predictor).
    hires_ch: int = 64
    window: int = 17
    attn_dim: int = 128
    num_blocks: int = 2
    num_refine_iters: int = 2
    max_offset_frac: float = 0.5
    # --- LiteShape (variant "lite_shape") only ---------------------------------
    # Single-resolution backbone + query-attention head + optional shape
    # projection. Reuses `backbone`, `feat_dim` (as the attention width),
    # `num_heads`, and the shape_* fields above. feature_index selects WHICH
    # single backbone stage is used (no multi-scale fusion); -2 is a mid/high-res
    # stage. Leave shape_basis_path empty to run as a pure coord regressor.
    feature_index: int = -2
    # --- DinoCoord (variant "dino_coord") only ---------------------------------
    # Frozen DINOv2 ViT backbone + query-attention head. Reuses `backbone`
    # (a timm dinov2 name), `feat_dim` (attention width), `num_heads`, and the
    # shape_* fields. freeze_backbone keeps DINOv2 frozen (small-data default);
    # dino_input is the patch-aligned square size fed to the ViT (multiple of 14).
    freeze_backbone: bool = True
    dino_input: int = 518


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 150
    batch_size: int = 32
    val_batch_size: int = 64
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    # LR multiplier for the graph-prior offset/Cholesky heads (graph_prior_fusion
    # only). These heads move pre-activations through bounded tanh/sigmoid, so at
    # the shared head LR they cannot travel far enough to change the prior's
    # behaviour within a training run. See TrainingEngine._split_prior_param_groups.
    lr_prior_mult: float = 10.0
    weight_decay: float = 1e-4
    lr_milestones: list = field(default_factory=lambda: [60, 90])
    lr_gamma: float = 0.1
    grad_clip: float = 0.5
    init_noise_sigma: float = 0.05
    init_mode: str = "mean"
    rot_factor: float = 0.0
    checkpoint_interval: int = 10
    vis_interval: int = 10
    seed: int = 42
    device: str = "cuda"
    loss_type: str = "mse"
    # Heatmap term used by heatmap_loss (non-WFLW heatmap + graph-cond family).
    # "ce" = cross entropy vs the normalized Gaussian target (default; actually
    # shapes the map). "mse" = legacy, contributes ~0.01% of the gradient and
    # cannot train the heatmap; kept only to reproduce earlier runs.
    heatmap_loss_mode: str = "ce"
    star_omega: float = 1.0
    star_eigenvalue_clamp: float = 6.0
    # Weight on the coarse-coord term for attn_local_refine. Keeps the frozen
    # Stage A anchor in the loss so Stage B's refinement is measured against it
    # rather than drifting. Stage A params are frozen, so this only shapes Stage B.
    coarse_loss_weight: float = 0.1
    # --- PIPNet (variant "pipnet") only ---------------------------------------
    # Composite-loss weights, matching the reference: cls (score-map MSE) is
    # weighted 10, and each of the four L1 offset terms is weighted 1. Total is
    # cls_w*loss_map + reg_w*(loss_x + loss_y + loss_nb_x + loss_nb_y).
    pipnet_cls_loss_weight: float = 10.0
    pipnet_reg_loss_weight: float = 1.0
    # Weight on the STAR coordinate term when model.pipnet_use_star is True.
    # star_omega and star_eigenvalue_clamp (defined above) are reused.
    pipnet_star_weight: float = 1.0
    # Weight on the STAR coordinate term when model.heatmap_use_star is True.
    heatmap_star_weight: float = 1.0
    # Coordinate-MSE weight in simcc_loss, alongside the two per-axis cross
    # entropies. Same default and same purpose as heatmap_loss's coord_weight:
    # it is what puts the windowed decoder in the gradient path, so the bins
    # cannot be well-classified while the readout stays miscalibrated.
    simcc_coord_weight: float = 100.0
    # Per-stage loss weights for hrnet_cascade intermediate supervision.
    # Empty list => equal weighting across all stages.
    cascade_stage_weights: list = field(default_factory=list)
    # Pose-invariant structural loss (edge-length + turning-angle) added to the
    # coordinate loss for coordinate-regression variants (fused_global family).
    # 0.0 disables it entirely (default). Keep modest so the coordinate loss still
    # leads (protects the well-localized across-edge component / the tail).
    structural_loss_weight: float = 0.0
    structural_dist_weight: float = 1.0
    structural_angle_weight: float = 1.0
    # Horizontal-flip augmentation probability for the Lizard dataset. The flip
    # relabels bilateral pairs (0<->1, 2<->3, 4<->5, 6<->7) so it is label-correct
    # for the toe/finger anatomy. 0.0 disables (previous behaviour); 0.5 is the
    # usual setting and roughly doubles effective data on the ~800-sample set.
    # Only applies when the dataset is built with augment=True (training).
    flip_prob: float = 0.5
    # Weight on the fused-posterior cross-entropy term for graph_prior_fusion.
    # The base heatmap CE trains only the APPEARANCE head; without this term the
    # graph prior is supervised solely by the weak coordinate MSE. This term
    # applies the same CE (vs the normalized Gaussian target) to the FUSED
    # log-posterior, giving the prior a distributional gradient. 0.0 disables it
    # (recovers the prior-via-coord-loss-only behaviour).
    fused_ce_weight: float = 1.0


@dataclass
class LandmarkingConfig:
    """Top-level configuration for the landmarking package."""

    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_json(cls, path: str) -> "LandmarkingConfig":
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON config file.

        Returns:
            Populated LandmarkingConfig instance.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "LandmarkingConfig":
        """Construct configuration from a nested dictionary.

        Args:
            data: Dictionary with optional keys: paths, dataset, model, training.

        Returns:
            Populated LandmarkingConfig instance.
        """
        paths = PathConfig(**data.get("paths", {}))
        dataset = DatasetConfig(**data.get("dataset", {}))
        model = ModelConfig(**data.get("model", {}))
        training = TrainingConfig(**data.get("training", {}))
        return cls(paths=paths, dataset=dataset, model=model, training=training)

    def resolve_paths(self) -> "LandmarkingConfig":
        """Apply environment variable overrides to path fields.

        Checks for LANDMARKING_{FIELD_NAME} environment variables and
        overrides corresponding fields. Also resolves dataset data_dir
        from data_root if not explicitly set.

        When landmark_indices is non-empty, auto-sets num_landmarks to
        the length of that list.

        Returns:
            self (mutated in place for convenience).
        """
        self.paths.data_root = resolve_path("DATA_ROOT", self.paths.data_root)
        self.paths.output_root = resolve_path("OUTPUT_ROOT", self.paths.output_root)
        self.paths.yolo_obb_model = resolve_path(
            "YOLO_OBB_MODEL", self.paths.yolo_obb_model
        )

        # Resolve dataset data_dir from data_root + dataset name if not set
        if not self.dataset.data_dir:
            self.dataset.data_dir = resolve_dataset_dir(
                self.paths.data_root, self.dataset.name
            )
        else:
            self.dataset.data_dir = resolve_path("DATA_DIR", self.dataset.data_dir)

        # Auto-set num_landmarks from landmark_indices when non-empty
        if self.dataset.landmark_indices:
            self.dataset.num_landmarks = len(self.dataset.landmark_indices)

        return self

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If landmark_indices contains invalid values or duplicates.
        """
        if self.dataset.landmark_indices:
            max_idx = 18 if self.dataset.name == "cephalometric" else 97
            for idx in self.dataset.landmark_indices:
                if not isinstance(idx, int) or idx < 0 or idx > max_idx:
                    raise ValueError(
                        f"Invalid landmark index: {idx}. Must be int in [0, {max_idx}]."
                    )
            if len(self.dataset.landmark_indices) != len(set(self.dataset.landmark_indices)):
                raise ValueError("landmark_indices contains duplicates")

    def to_dict(self) -> dict:
        """Serialize configuration to a dictionary."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Write configuration to a JSON file.

        Args:
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
