"""Unified training engine for landmark detection.

Based on the canonical train_wflw.py training loop. Supports all model
variants via the registry, both datasets, differential LR, gradient
clipping, MultiStepLR, noise initialization, rotation augmentation,
checkpoint saving, and overlay visualization.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config.schema import LandmarkingConfig
from ..common.graph_topologies import get_edge_index
from ..models.registry import get_model
from .loss import landmark_loss
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

        # Build edge index
        self.edge_index = get_edge_index(
            cfg.dataset.graph_topology, cfg.dataset.num_landmarks
        ).to(self.device)

        # Load mean shape if specified
        if cfg.dataset.mean_shape_path and Path(cfg.dataset.mean_shape_path).exists():
            self.mean_shape = torch.load(
                cfg.dataset.mean_shape_path, map_location=self.device, weights_only=False
            )
            logging.info(f"Loaded mean shape: {self.mean_shape.shape}")

        # Instantiate model from registry
        model_kwargs = {
            "num_landmarks": cfg.dataset.num_landmarks,
            "feat_dim": cfg.model.feat_dim,
            "gnn_hidden": cfg.model.gnn_hidden,
            "num_layers": cfg.model.num_layers,
            "num_iters": cfg.model.num_iters,
        }
        # Add variant-specific kwargs
        if cfg.model.variant in ("multiscale", "fused"):
            model_kwargs["scale_indices"] = cfg.model.scale_indices
        if cfg.model.variant == "coord":
            model_kwargs["use_coarse_init"] = cfg.model.use_coarse_init
        if cfg.model.variant == "hinit":
            model_kwargs["heatmap_checkpoint"] = cfg.model.heatmap_checkpoint

        self.model = get_model(cfg.model.variant, **model_kwargs)
        self.model.to(self.device)

        # Load backbone pretrained weights if specified
        if cfg.model.backbone_pretrained_path:
            self._load_backbone_weights(cfg.model.backbone_pretrained_path)

        # Optimizer with differential LR
        param_groups = make_param_groups(
            self.model, cfg.training.lr, cfg.training.lr_backbone
        )
        self.optimizer = torch.optim.Adam(param_groups)

        # MultiStepLR scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=cfg.training.lr_milestones,
            gamma=cfg.training.lr_gamma,
        )

        logging.info(
            f"Model: {cfg.model.variant}, "
            f"LR: backbone={cfg.training.lr_backbone}, head={cfg.training.lr}, "
            f"Milestones: {cfg.training.lr_milestones}"
        )

    def train(self):
        """Main training loop.

        Iterates over epochs, calling _train_epoch and _validate,
        saving checkpoints and overlays at configured intervals.
        """
        cfg = self.config
        best_val = float("inf")

        for epoch in range(1, cfg.training.epochs + 1):
            epoch_loss = self._train_epoch(epoch)
            metrics = self._validate(epoch)
            val_loss = metrics.get("val_loss", float("inf"))

            logging.info(
                f"Epoch {epoch}/{cfg.training.epochs}, "
                f"Train Loss: {epoch_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}"
            )

            self.scheduler.step()

            # Save checkpoint
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
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
            imgs, coords, *rest = batch
            imgs = imgs.to(self.device)
            coords = coords.to(self.device)
            B = imgs.shape[0]

            # Build initial coordinates
            initial_coords = self._get_initial_coords(B, coords, epoch)

            # Forward pass
            out = self.model(imgs, initial_coords, self.edge_index)
            if isinstance(out, tuple):
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
            Dict with validation metrics.
        """
        self.model.eval()
        cfg = self.config
        val_loss_total = 0.0
        n_samples = 0
        self._last_vis_data = None

        with torch.no_grad():
            for batch in self.val_loader:
                imgs, coords, *rest = batch
                imgs = imgs.to(self.device)
                coords = coords.to(self.device)
                B = imgs.shape[0]

                # Use mean shape for validation
                initial_coords = self._get_initial_coords(B, coords, epoch)

                out = self.model(imgs, initial_coords, self.edge_index)
                pred_coords = out[0] if isinstance(out, tuple) else out

                val_loss_total += landmark_loss(pred_coords, coords).item() * B
                n_samples += B

                # Store last batch for visualization
                if self._last_vis_data is None:
                    self._last_vis_data = (
                        imgs.detach(),
                        pred_coords.detach(),
                        coords.detach(),
                    )

        val_loss = val_loss_total / max(n_samples, 1)
        return {"val_loss": val_loss}

    def _get_initial_coords(
        self, batch_size: int, gt_coords: torch.Tensor, epoch: int
    ) -> torch.Tensor:
        """Generate initial coordinate estimates for the GCN.

        Supports mean shape initialization (with noise) or
        ground-truth + noise initialization.
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

        # Default: mean shape + noise
        if self.mean_shape is not None:
            ms = self.mean_shape.unsqueeze(0).expand(batch_size, -1, -1)
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

        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
        """
        ckpt_dir = Path(self.output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        if is_best:
            torch.save(state, str(ckpt_dir / "best.pth"))

        torch.save(state, str(ckpt_dir / f"epoch_{epoch:04d}.pth"))

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
