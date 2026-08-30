"""Unified training engine for landmark detection.

Based on the canonical train_wflw.py training loop. Supports all model
variants via the registry, both datasets, differential LR, gradient
clipping, MultiStepLR, noise initialization, rotation augmentation,
checkpoint saving, and overlay visualization.
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config.schema import LandmarkingConfig
from ..common.graph_topologies import get_edge_index
from ..models.registry import get_model
from .loss import landmark_loss, heatmap_loss, star_loss
from .utils import set_seed, get_device, make_param_groups, make_output_dir
from .visualization import save_training_overlays

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class TrainingEngine:
    """Unified training loop for landmark detection models.

    Supports all model variants, both datasets (Lizard and WFLW),
    differential learning rates, gradient clipping, MultiStepLR scheduling,
    noise-augmented initialization, and checkpoint/overlay management.

    Args:
        config: LandmarkingConfig instance with all training parameters.
    """

    def __init__(self, config: LandmarkingConfig):
        self.config = config
        self.device = get_device(config.training.device)
        self.output_dir = make_output_dir(
            config.paths.output_root,
            config.dataset.name,
            config.model.variant,
        )
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.mean_shape = None
        self.edge_index = None

    def setup(self):
        """Initialize model, optimizer, scheduler, and dataloaders.

        Must be called before train(). Sets seed, builds model from
        registry, configures optimizer with differential LR, creates
        data loaders.
        """
        cfg = self.config
        set_seed(cfg.training.seed)

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        ckpt_dir = Path(self.output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save resolved config for reproducibility
        cfg.to_json(str(Path(self.output_dir) / "config.json"))

        # Sparsity: when landmark_indices is set, keep wflw topology
        # (the subsampled version preserves anatomical groupings)
        landmark_indices = cfg.dataset.landmark_indices

        # Build edge index — pass landmark_indices for subsampled WFLW graph
        self.edge_index = get_edge_index(
            cfg.dataset.graph_topology, cfg.dataset.num_landmarks,
            landmark_indices=landmark_indices or None,
        ).to(self.device)

        # Load mean shape if specified
        if cfg.dataset.mean_shape_path and Path(cfg.dataset.mean_shape_path).exists():
            self.mean_shape = torch.load(
                cfg.dataset.mean_shape_path, map_location=self.device, weights_only=False
            )
            logging.info(f"Loaded mean shape: {self.mean_shape.shape}")
        elif cfg.dataset.mean_shape_path:
            logging.warning(
                f"mean_shape_path specified but file not found: {cfg.dataset.mean_shape_path}. "
                f"Falling back to constant (0.5, 0.5) initialization."
            )
        elif cfg.model.variant not in ("heatmap", "hrnet_coord", "stacked_hourglass", "vit", "graph_cond_heatmap", "graph_prior_fusion"):
            logging.warning(
                "No mean_shape_path set for GCN model. Using constant (0.5, 0.5) initialization. "
                "This will likely result in very poor initial NME. "
                "Generate a mean shape with: python -m landmarking.datasets.wflw.compute_mean_shape"
            )

        # Sparsity: subsample mean_shape to match landmark_indices
        if self.mean_shape is not None and landmark_indices:
            self.mean_shape = self.mean_shape[landmark_indices]

        # Compute flipped mean shape for flip-aware GCN initialization
        self.mean_shape_flipped = None
        if self.mean_shape is not None and cfg.dataset.name == "wflw":
            from .flip_utils import compute_flipped_mean_shape
            self.mean_shape_flipped = compute_flipped_mean_shape(
                self.mean_shape, cfg.dataset.num_landmarks, landmark_indices or None
            )

        # Determine if this is a coord-only model (image-only forward, no edge_index)
        self._is_coord_only_model = cfg.model.variant in ("hrnet_coord",)

        # Determine if this is a STAR loss model (returns coords + log_sigma)
        self._is_star_model = cfg.model.variant in ("fused_global_star",)
        self._use_star_loss = cfg.training.loss_type == "star" or self._is_star_model

        # Instantiate model from registry
        model_kwargs = {
            "num_landmarks": cfg.dataset.num_landmarks,
        }
        # GCN-specific kwargs
        if cfg.model.variant not in ("heatmap", "hrnet_coord", "stacked_hourglass", "vit", "graph_cond_heatmap", "graph_prior_fusion"):
            model_kwargs.update({
                "feat_dim": cfg.model.feat_dim,
                "gnn_hidden": cfg.model.gnn_hidden,
                "num_layers": cfg.model.num_layers,
                "num_iters": cfg.model.num_iters,
            })
        # Add variant-specific kwargs
        if cfg.model.variant in ("multiscale", "fused", "fused_global_ms"):
            model_kwargs["scale_indices"] = cfg.model.scale_indices
        if cfg.model.variant == "fused_global_patch":
            model_kwargs.update({
                "patch_mode": cfg.model.patch_mode,
                "patch_step": cfg.model.patch_step,
                "patch_radius": cfg.model.patch_radius,
                "patch_radii": tuple(cfg.model.patch_radii),
                "patch_proj_dim": cfg.model.patch_proj_dim,
            })
        if cfg.model.variant == "coord":
            model_kwargs["use_coarse_init"] = cfg.model.use_coarse_init
        if cfg.model.variant == "hinit":
            model_kwargs["heatmap_checkpoint"] = cfg.model.heatmap_checkpoint
        if cfg.model.variant == "heatmap":
            model_kwargs["heatmap_size"] = getattr(cfg.model, "heatmap_size", 64)
        if cfg.model.variant == "graph_cond_heatmap":
            model_kwargs.update({
                "gnn_hidden": cfg.model.gnn_hidden,
                "num_layers": cfg.model.num_layers,
                "num_heads": getattr(cfg.model, "num_heads", 4),
                "heatmap_size": cfg.model.heatmap_size,
            })
        if cfg.model.variant == "graph_prior_fusion":
            model_kwargs.update({
                "gnn_hidden": cfg.model.gnn_hidden,
                "num_layers": cfg.model.num_layers,
                "heatmap_size": cfg.model.heatmap_size,
                "sigma_min": cfg.model.prior_sigma_min,
                "sigma_span": cfg.model.prior_sigma_span,
                "offset_scale": cfg.model.prior_offset_scale,
                "chol_bias": getattr(cfg.model, "prior_chol_bias", 4.0),
                "prior_disabled": cfg.model.prior_disabled,
            })

        # Determine if this is a heatmap-style model (different forward signature)
        # Only True for WFLW heatmap variant which uses WFLWRefDataset with pre-generated target heatmaps
        self._is_heatmap_model = (
            cfg.model.variant in ("heatmap",) and cfg.dataset.name == "wflw"
        )
        # Heatmap on Lizard: uses standard dataset + heatmap_loss (same path as graph_cond_heatmap)
        self._is_heatmap_on_coords = (
            cfg.model.variant in ("heatmap",) and cfg.dataset.name != "wflw"
        )
        # Graph-conditioned heatmap: hybrid (edge_index input + heatmap output).
        # graph_prior_fusion shares the identical forward(imgs, edge_index) ->
        # (heatmaps, coords) interface, so it uses the same dispatch/loss path.
        self._is_graph_cond_heatmap = cfg.model.variant in (
            "graph_cond_heatmap", "graph_prior_fusion",
        )
        # graph_prior_fusion needs two extras the shared path doesn't: a fusion
        # warm-up toggle and Sigma monitoring for collapse detection.
        self._is_graph_prior_fusion = cfg.model.variant == "graph_prior_fusion"

        self.model = get_model(cfg.model.variant, **model_kwargs)
        self.model.to(self.device)

        # Load backbone pretrained weights if specified
        if cfg.model.backbone_pretrained_path:
            self._load_backbone_weights(cfg.model.backbone_pretrained_path)

        # Optimizer — heatmap variant uses single Adam (all params same LR),
        # other variants use differential LR for backbone vs head.
        if self._is_heatmap_model:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
            )
        else:
            param_groups = make_param_groups(
                self.model, cfg.training.lr, cfg.training.lr_backbone
            )
            if self._is_graph_prior_fusion:
                param_groups = self._split_prior_param_groups(param_groups)
            self.optimizer = torch.optim.Adam(param_groups)

        # MultiStepLR scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=cfg.training.lr_milestones,
            gamma=cfg.training.lr_gamma,
        )

        # For heatmap variant, suppress the scheduler.step() before optimizer.step()
        # warning — this is intentional to match reference tools/train.py behaviour.
        if self._is_heatmap_model:
            warnings.filterwarnings(
                "ignore",
                message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
                category=UserWarning,
            )

        logging.info(
            f"Model: {cfg.model.variant}, "
            f"LR: backbone={cfg.training.lr_backbone}, head={cfg.training.lr}, "
            f"Milestones: {cfg.training.lr_milestones}"
        )

        # Create dataloaders from split
        self._create_dataloaders()

    def _split_prior_param_groups(self, param_groups: list) -> list:
        """Give the graph-prior heads (offset/Cholesky) their own faster LR group.

        Why this is needed: these two heads emit the prior's mean offset and its
        breadth through bounded squashing functions (tanh / sigmoid), so what the
        optimizer actually moves is a PRE-ACTIVATION. With Adam, a parameter
        travels roughly lr * num_steps when gradients are consistent. On a small
        dataset that budget is well under one unit of pre-activation, which is far
        less than the distance between "prior effectively off" and "prior
        informative". Left on the shared head LR, the prior is frozen at whatever
        breadth it was initialized with, and the architecture silently degenerates
        to a pure heatmap model — a false negative for the whole research question.
        """
        cfg = self.config
        mult = getattr(cfg.training, "lr_prior_mult", 10.0)
        prior_modules = [self.model.offset_head, self.model.chol_head]
        prior_params = [p for m in prior_modules for p in m.parameters()]
        prior_ids = {id(p) for p in prior_params}

        new_groups = []
        for g in param_groups:
            kept = [p for p in g["params"] if id(p) not in prior_ids]
            if kept:
                new_groups.append({**g, "params": kept})
        new_groups.append({
            "params": prior_params,
            "lr": cfg.training.lr * mult,
            "weight_decay": 0.0,
        })
        logging.info(
            f"Graph-prior heads on dedicated LR "
            f"{cfg.training.lr * mult:.2e} ({mult}x head LR)"
        )
        return new_groups

    def _create_dataloaders(self):
        """Create train and validation dataloaders from split file.

        If no split_path is configured, auto-discovers .pt files from data_dir
        and uses an 80/20 train/val split.
        """
        import json as _json
        from pathlib import Path as _Path

        cfg = self.config
        dataset_name = cfg.dataset.name

        # Load split or auto-discover
        if cfg.dataset.split_path and _Path(cfg.dataset.split_path).exists():
            with open(cfg.dataset.split_path) as f:
                split_data = _json.load(f)
            train_paths = split_data.get("train", [])
            val_paths = split_data.get("val", [])
        else:
            # Auto-discover from data_dir/pt_crops/train (WFLW) or data_dir/train (Lizard)
            data_dir = _Path(cfg.dataset.data_dir)
            candidates = [
                data_dir / "pt_crops" / "train",  # WFLW layout
                data_dir / "train",                # Lizard layout
            ]
            pt_dir = None
            for c in candidates:
                if c.exists():
                    pt_dir = c
                    break
            if pt_dir is None:
                raise FileNotFoundError(
                    f"No training data found. Searched: {[str(c) for c in candidates]}. "
                    f"Run preprocessing first, or set dataset.split_path in config."
                )
            all_paths = sorted([str(p) for p in pt_dir.glob("*.pt")])
            if not all_paths:
                raise FileNotFoundError(f"No .pt files found in {pt_dir}")

            # 80/20 train/val split from the training pool
            import random
            rng = random.Random(cfg.training.seed)
            rng.shuffle(all_paths)
            split_idx = int(len(all_paths) * 0.8)
            train_paths = sorted(all_paths[:split_idx])
            val_paths = sorted(all_paths[split_idx:])
            logging.info(
                f"Auto-split: {len(train_paths)} train, {len(val_paths)} val "
                f"from {pt_dir}"
            )

            # For WFLW: always include the official test set from pt_crops/test/
            if dataset_name == "wflw":
                test_dir = data_dir / "pt_crops" / "test"
                if test_dir.exists():
                    test_paths = sorted([str(p) for p in test_dir.glob("*.pt")])
                    if test_paths:
                        logging.info(
                            f"Official WFLW test set: {len(test_paths)} samples "
                            f"from {test_dir}"
                        )
                        # Store for evaluation access
                        self._test_paths = test_paths

        if not train_paths:
            raise ValueError("Train set is empty. Check split file or data directory.")
        if not val_paths:
            raise ValueError("Val set is empty. Check split file or data directory.")

        logging.info(f"Train: {len(train_paths)} samples, Val: {len(val_paths)} samples")

        # Create dataset instances
        if dataset_name == "wflw":
            if cfg.model.variant == "heatmap":
                # Use the reference HRNet dataset for paper-matching results
                from ..datasets.wflw.dataset_ref import WFLWRefDataset

                train_ds = WFLWRefDataset(
                    pt_paths=train_paths,
                    augment=True,
                    landmark_indices=cfg.dataset.landmark_indices or None,
                )
                val_ds = WFLWRefDataset(
                    pt_paths=val_paths,
                    augment=False,
                    landmark_indices=cfg.dataset.landmark_indices or None,
                )
            else:
                from ..datasets.wflw.dataset import WFLWDataset

                train_ds = WFLWDataset(
                    pt_paths=train_paths,
                    input_size=cfg.dataset.input_size,
                    num_landmarks=cfg.dataset.num_landmarks,
                    augment=True,
                    rot_factor=cfg.training.rot_factor,
                    landmark_indices=cfg.dataset.landmark_indices or None,
                )
                val_ds = WFLWDataset(
                    pt_paths=val_paths,
                    input_size=cfg.dataset.input_size,
                    num_landmarks=cfg.dataset.num_landmarks,
                    augment=False,
                    landmark_indices=cfg.dataset.landmark_indices or None,
                )
        elif dataset_name == "cephalometric":
            from ..datasets.cephalometric.dataset import CephalometricDataset

            ceph_mode = "heatmap" if cfg.model.variant == "heatmap" else "coord"
            train_ds = CephalometricDataset(
                pt_paths=train_paths,
                input_size=cfg.dataset.input_size,
                num_landmarks=cfg.dataset.num_landmarks,
                augment=True,
                mode=ceph_mode,
                heatmap_size=cfg.model.heatmap_size,
                sigma=cfg.model.sigma,
                pixel_spacing=cfg.dataset.pixel_spacing,
                landmark_indices=cfg.dataset.landmark_indices or None,
                split="train",
            )
            val_ds = CephalometricDataset(
                pt_paths=val_paths,
                input_size=cfg.dataset.input_size,
                num_landmarks=cfg.dataset.num_landmarks,
                augment=False,
                mode=ceph_mode,
                heatmap_size=cfg.model.heatmap_size,
                sigma=cfg.model.sigma,
                pixel_spacing=cfg.dataset.pixel_spacing,
                landmark_indices=cfg.dataset.landmark_indices or None,
                split="test1",
            )
        else:
            # Lizard or generic
            from ..datasets.lizard.dataset import LizardDataset

            train_ds = LizardDataset(
                pt_paths=train_paths,
                input_size=cfg.dataset.input_size,
                num_landmarks=cfg.dataset.num_landmarks,
                augment=True,
            )
            val_ds = LizardDataset(
                pt_paths=val_paths,
                input_size=cfg.dataset.input_size,
                num_landmarks=cfg.dataset.num_landmarks,
                augment=False,
            )

        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.training.val_batch_size, shuffle=False, num_workers=4
        )

    def train(self):
        """Main training loop.

        Iterates over epochs, calling _train_epoch and _validate,
        saving checkpoints and overlays at configured intervals.

        For heatmap variant: scheduler.step() at epoch START (before training),
        best checkpoint based on val NME.
        For other variants: scheduler.step() at epoch END, best checkpoint based
        on val loss.
        """
        cfg = self.config
        best_val = float("inf")
        best_nme = float("inf")

        for epoch in range(1, cfg.training.epochs + 1):
            # Heatmap variant: step LR at epoch start (matches reference)
            if self._is_heatmap_model:
                self.scheduler.step()

            # Fusion warm-up: run as a pure heatmap model for the first N epochs so
            # the appearance head learns real peaks before the graph prior starts
            # acting on anchors derived from it.
            if self._is_graph_prior_fusion:
                warmup = getattr(cfg.model, "prior_warmup_epochs", 0)
                prior_on = epoch > warmup
                self.model.set_prior_active(prior_on)
                if warmup and epoch == warmup + 1:
                    logging.info(
                        f"Fusion warm-up complete after {warmup} epochs; "
                        f"graph prior now ACTIVE."
                    )

            epoch_loss = self._train_epoch(epoch)
            metrics = self._validate(epoch)
            val_loss = metrics.get("val_loss", float("inf"))
            val_nme = metrics.get("val_nme", None)

            # Sigma-collapse monitoring: trace pinned near the ceiling means the
            # prior has effectively switched itself off.
            if self._is_graph_prior_fusion:
                trace = getattr(self.model, "last_sigma_trace", None)
                if trace is not None:
                    logging.info(
                        f"Epoch {epoch} prior sigma trace: {float(trace):.6f} "
                        f"(prior_active={self.model.prior_active})"
                    )

            if self._is_heatmap_model and val_nme is not None:
                logging.info(
                    f"Epoch {epoch}/{cfg.training.epochs}, "
                    f"Train Loss: {epoch_loss:.6f}, "
                    f"Val NME: {val_nme:.4f}, "
                    f"Val Pixel Error (512px): {metrics.get('val_px_err', 0):.2f}"
                )
                # Save best based on NME
                is_best = val_nme < best_nme
                if is_best:
                    best_nme = val_nme
            elif val_nme is not None:
                logging.info(
                    f"Epoch {epoch}/{cfg.training.epochs}, "
                    f"Train Loss: {epoch_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"Val NME: {val_nme:.4f}, "
                    f"Val Pixel Error ({cfg.dataset.input_size}px): "
                    f"{metrics.get('val_px_err', 0):.2f}"
                )
                # Save best based on val loss (GCN/coord)
                is_best = val_loss < best_val
                if is_best:
                    best_val = val_loss
            else:
                logging.info(
                    f"Epoch {epoch}/{cfg.training.epochs}, "
                    f"Train Loss: {epoch_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"Val Pixel Error ({cfg.dataset.input_size}px): "
                    f"{metrics.get('val_px_err', 0):.2f}"
                )
                # Save best based on val loss
                is_best = val_loss < best_val
                if is_best:
                    best_val = val_loss

            # Non-heatmap: step scheduler at end
            if not self._is_heatmap_model:
                self.scheduler.step()

            # Save checkpoint
            if is_best or epoch % cfg.training.checkpoint_interval == 0:
                self._save_checkpoint(epoch, is_best)

            # Save overlays
            if epoch <= 20 or epoch % cfg.training.vis_interval == 0:
                if hasattr(self, "_last_vis_data") and self._last_vis_data is not None:
                    imgs, pred, gt = self._last_vis_data
                    save_training_overlays(
                        imgs, pred, gt, self.output_dir, epoch,
                        input_size=cfg.dataset.input_size,
                    )

    def _train_epoch(self, epoch: int) -> float:
        """Single training epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        cfg = self.config
        epoch_loss = 0.0
        n_samples = 0

        for batch in self.train_loader:
            if self._is_heatmap_model:
                # Reference heatmap pipeline: (img, target_hm, meta)
                imgs, target_hm, meta = batch
                imgs = imgs.to(self.device)
                target_hm = target_hm.to(self.device)
                B = imgs.shape[0]

                pred_heatmaps, _ = self.model(imgs)
                # Pure heatmap MSE loss — no coordinate loss
                loss = torch.nn.functional.mse_loss(pred_heatmaps, target_hm)
            elif self._is_graph_cond_heatmap:
                # Graph-conditioned heatmap: forward(imgs, edge_index) → (heatmaps, coords)
                imgs, coords, *rest = batch
                imgs = imgs.to(self.device)
                coords = coords.to(self.device)
                B = imgs.shape[0]

                pred_heatmaps, pred_coords = self.model(imgs, self.edge_index)
                loss = heatmap_loss(
                    pred_heatmaps, pred_coords, coords,
                    cfg.model.heatmap_size, cfg.model.sigma,
                )
            elif self._is_heatmap_on_coords:
                # Heatmap model on non-WFLW dataset: forward(imgs) → (heatmaps, coords)
                imgs, coords, *rest = batch
                imgs = imgs.to(self.device)
                coords = coords.to(self.device)
                B = imgs.shape[0]

                pred_heatmaps, pred_coords = self.model(imgs)
                loss = heatmap_loss(
                    pred_heatmaps, pred_coords, coords,
                    cfg.model.heatmap_size, cfg.model.sigma,
                )
            elif self._is_coord_only_model:
                # Coordinate-only model (e.g. hrnet_coord): forward(imgs) → (B, N, 2)
                imgs, coords, *rest = batch
                imgs = imgs.to(self.device)
                coords = coords.to(self.device)
                B = imgs.shape[0]

                pred_coords = self.model(imgs)
                loss = landmark_loss(pred_coords, coords)
            else:
                imgs, coords, *rest = batch
                imgs = imgs.to(self.device)
                coords = coords.to(self.device)
                B = imgs.shape[0]

                # Extract flip flag from metadata for flip-aware init
                flipped = None
                if rest and isinstance(rest[0], dict) and "was_flipped" in rest[0]:
                    flipped = rest[0]["was_flipped"]
                elif rest and hasattr(rest[0], "get"):
                    flipped = rest[0].get("was_flipped")

                # GCN model: forward(x, initial_coords, edge_index) → coords
                initial_coords = self._get_initial_coords(B, coords, epoch, flipped=flipped)
                out = self.model(imgs, initial_coords, self.edge_index)
                if isinstance(out, tuple) and len(out) == 2 and self._use_star_loss:
                    # STAR model with intermediate supervision:
                    # Training: out = (all_coords_list, log_sigma)
                    # Eval: out = (final_coords, log_sigma)
                    first_elem, log_sigma = out
                    if isinstance(first_elem, list):
                        # Intermediate supervision: MSE on early iters, STAR on final
                        all_coords_list = first_elem
                        num_iters = len(all_coords_list)
                        # Linearly increasing weights for intermediate MSE
                        inter_weights = [
                            (i + 1) / num_iters * 0.5
                            for i in range(num_iters - 1)
                        ]
                        loss = sum(
                            w * landmark_loss(c, coords)
                            for w, c in zip(inter_weights, all_coords_list[:-1])
                        )
                        # STAR loss on final iteration (full weight)
                        pred_coords = all_coords_list[-1]
                        loss = loss + star_loss(
                            pred_coords, coords, log_sigma,
                            omega=cfg.training.star_omega,
                            eigenvalue_clamp=cfg.training.star_eigenvalue_clamp,
                        )
                    else:
                        # Eval mode or non-list: single coords tensor
                        pred_coords = first_elem
                        loss = star_loss(
                            pred_coords, coords, log_sigma,
                            omega=cfg.training.star_omega,
                            eigenvalue_clamp=cfg.training.star_eigenvalue_clamp,
                        )
                elif isinstance(out, tuple):
                    pred_coords = out[0]
                    loss = landmark_loss(pred_coords, coords)
                else:
                    pred_coords = out
                    loss = landmark_loss(pred_coords, coords)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.training.grad_clip
            )
            self.optimizer.step()

            epoch_loss += loss.item() * B
            n_samples += B

        return epoch_loss / max(n_samples, 1)

    def _validate(self, epoch: int) -> dict:
        """Validation pass.

        Args:
            epoch: Current epoch number.

        Returns:
            Dict with validation metrics. For heatmap variant, includes
            val_nme and val_px_err. For other variants, includes val_loss.
        """
        self.model.eval()
        cfg = self.config
        self._last_vis_data = None

        if self._is_heatmap_model:
            return self._validate_heatmap(epoch)

        # Standard coordinate-based validation
        from ..evaluation.metrics_wflw import compute_nme as compute_nme_metrics, get_iod_indices_in_subset

        val_loss_total = 0.0
        n_samples = 0
        nme_sum = 0.0
        nme_count = 0
        px_err_sum = 0.0

        # Determine IOD positions for NME
        landmark_indices = cfg.dataset.landmark_indices
        if landmark_indices:
            iod_left, iod_right = get_iod_indices_in_subset(landmark_indices)
        elif cfg.dataset.name == "wflw":
            iod_left, iod_right = 60, 72
        else:
            iod_left, iod_right = None, None  # Lizard: no NME

        with torch.no_grad():
            for batch in self.val_loader:
                imgs, coords, *rest = batch
                imgs = imgs.to(self.device)
                coords = coords.to(self.device)
                B = imgs.shape[0]

                # Extract flip flag from metadata
                flipped = None
                if rest and isinstance(rest[0], dict) and "was_flipped" in rest[0]:
                    flipped = rest[0]["was_flipped"]
                elif rest and hasattr(rest[0], "get"):
                    flipped = rest[0].get("was_flipped")

                if self._is_graph_cond_heatmap:
                    _, pred_coords = self.model(imgs, self.edge_index)
                elif self._is_heatmap_on_coords:
                    _, pred_coords = self.model(imgs)
                elif self._is_coord_only_model:
                    pred_coords = self.model(imgs)
                else:
                    initial_coords = self._get_initial_coords(B, coords, epoch, flipped=flipped)
                    out = self.model(imgs, initial_coords, self.edge_index)
                    pred_coords = out[0] if isinstance(out, tuple) else out
                val_loss_total += landmark_loss(pred_coords, coords).item() * B

                # Compute pixel error (all datasets)
                pred_px = pred_coords.cpu().numpy() * cfg.dataset.input_size
                gt_px = coords.cpu().numpy() * cfg.dataset.input_size
                px_err_sum += float(
                    ((pred_coords - coords).norm(dim=-1).mean(dim=-1).sum().item())
                    * cfg.dataset.input_size
                )

                # Compute NME per sample (WFLW only)
                if iod_left is not None and iod_right is not None:
                    pred_px = pred_coords.cpu().numpy() * 512.0
                    gt_px = coords.cpu().numpy() * 512.0
                    for i in range(B):
                        nme = compute_nme_metrics(pred_px[i], gt_px[i], iod_left=iod_left, iod_right=iod_right)
                        if nme is not None:
                            nme_sum += nme
                            nme_count += 1

                n_samples += B

                # Store last batch for visualization
                if self._last_vis_data is None:
                    self._last_vis_data = (
                        imgs.detach(),
                        pred_coords.detach(),
                        coords.detach(),
                    )

        val_loss = val_loss_total / max(n_samples, 1)
        result = {"val_loss": val_loss}
        result["val_px_err"] = px_err_sum / max(n_samples, 1)
        if nme_count > 0:
            result["val_nme"] = nme_sum / nme_count
        return result

    def _validate_heatmap(self, epoch: int) -> dict:
        """Heatmap-specific validation with NME evaluation.

        Uses decode_preds + compute_nme from the local evaluation module
        to compute paper-matching NME in 512px space.

        Returns:
            Dict with val_nme and val_px_err.
        """
        from ..evaluation.decode_preds import decode_preds, compute_nme
        from ..models.hrnet_heatmap import hard_argmax

        cfg = self.config
        heatmap_size = [
            getattr(cfg.model, "heatmap_size", 64),
            getattr(cfg.model, "heatmap_size", 64),
        ]
        nme_sum = 0.0
        nme_count = 0
        px_sum = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                imgs, target_hm, meta = batch
                imgs = imgs.to(self.device)
                B = imgs.shape[0]

                pred_hm, _ = self.model(imgs)
                score_map = pred_hm.cpu()

                # decode_preds: argmax + sub-pixel refinement + inverse affine
                preds = decode_preds(
                    score_map,
                    meta["center"],
                    meta["scale"],
                    heatmap_size,
                )  # (B, 98, 2) in 512px space

                # compute_nme: normalises by inter-ocular distance
                # For subsets, pass IOD positions explicitly
                landmark_indices = cfg.dataset.landmark_indices
                if landmark_indices:
                    from ..evaluation.metrics_wflw import get_iod_indices_in_subset
                    iod_l, iod_r = get_iod_indices_in_subset(landmark_indices)
                    nme_batch = compute_nme(preds, meta, iod_left=iod_l, iod_right=iod_r)
                else:
                    nme_batch = compute_nme(preds, meta)  # (B,) per-sample NME
                nme_sum += nme_batch.sum()
                nme_count += B

                # Also track pixel error in 512px space
                coords_norm = hard_argmax(pred_hm)  # (B, 98, 2) in [0,1]
                gt_norm = meta["pts"].to(self.device) / 512.0
                px_sum += (
                    (coords_norm - gt_norm).norm(dim=-1)
                    .mean(dim=-1).sum().item() * 512
                )

                # Store last batch for visualization
                if self._last_vis_data is None:
                    self._last_vis_data = (
                        imgs.detach(),
                        coords_norm.detach(),
                        gt_norm.detach(),
                    )

        nme_avg = nme_sum / max(nme_count, 1)
        px_avg = px_sum / max(nme_count, 1)
        return {"val_nme": nme_avg, "val_px_err": px_avg, "val_loss": nme_avg}

    def _get_initial_coords(
        self, batch_size: int, gt_coords: torch.Tensor, epoch: int,
        flipped: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate initial coordinate estimates for the GCN.

        Supports mean shape initialization (with noise) or
        ground-truth + noise initialization.

        When flipped is provided and mean_shape_flipped exists, uses the
        flipped mean shape for horizontally-flipped samples. This matches
        the reference train_wflw.py behavior.
        """
        cfg = self.config

        if cfg.training.init_mode == "gt_noise":
            noise = (
                torch.randn(
                    batch_size, cfg.dataset.num_landmarks, 2, device=self.device
                )
                * cfg.training.init_noise_sigma
            )
            return gt_coords + noise

        # Default: mean shape + noise, flip-aware
        if self.mean_shape is not None:
            ms_base = self.mean_shape.unsqueeze(0).expand(batch_size, -1, -1)

            # Per-sample flip-aware initialization
            if flipped is not None and self.mean_shape_flipped is not None:
                ms_flip = self.mean_shape_flipped.unsqueeze(0).expand(batch_size, -1, -1)
                flip_mask = flipped.to(self.device).view(batch_size, 1, 1).float()
                ms = ms_flip * flip_mask + ms_base * (1.0 - flip_mask)
            else:
                ms = ms_base
        else:
            ms = torch.full(
                (batch_size, cfg.dataset.num_landmarks, 2),
                0.5,
                device=self.device,
            )

        noise = (
            torch.randn(
                batch_size, cfg.dataset.num_landmarks, 2, device=self.device
            )
            * cfg.training.init_noise_sigma
        )
        return ms + noise

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        """Save model checkpoint.

        Only saves the best checkpoint to conserve disk space.

        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
        """
        if not is_best:
            return

        ckpt_dir = Path(self.output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(state, str(ckpt_dir / "best.pth"))

    def _load_backbone_weights(self, path: str) -> None:
        """Load backbone weights from an external checkpoint."""
        ext_state = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ext_state, dict) and "state_dict" in ext_state:
            ext_state = ext_state["state_dict"]

        backbone_state = self.model.backbone.state_dict()
        matched = {}
        for k, v in ext_state.items():
            if k.startswith("backbone."):
                inner_key = k[len("backbone."):]
                if (
                    inner_key in backbone_state
                    and backbone_state[inner_key].shape == v.shape
                ):
                    matched[inner_key] = v

        backbone_state.update(matched)
        self.model.backbone.load_state_dict(backbone_state, strict=True)
        logging.info(
            f"Backbone weights loaded: {len(matched)}/{len(backbone_state)} keys matched"
        )
