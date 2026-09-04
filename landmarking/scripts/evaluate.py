"""CLI tool for evaluating a trained landmark detection model.

Loads a checkpoint, runs inference on the test split, and outputs
dataset-appropriate metrics:
  - WFLW: NME, FR@0.1, AUC@0.1 (full + per-attribute subsets)
  - Lizard: pixel error, mm error (per-landmark)

Usage:
    # Evaluate GCN on WFLW
    python -m landmarking.scripts.evaluate \
        --config landmarking/config/defaults/wflw.json \
        --checkpoint runs/wflw/fused_global/.../checkpoints/best.pth \
        --split runs/wflw/splits/wflw_split.json \
        --output runs/wflw/fused_global/eval.json

    # Evaluate heatmap on WFLW (uses reference decode_preds)
    python -m landmarking.scripts.evaluate \
        --config landmarking/config/defaults/wflw_heatmap.json \
        --checkpoint runs/wflw/heatmap/.../checkpoints/best.pth \
        --split runs/wflw/splits/wflw_split.json \
        --output runs/wflw/heatmap/eval.json

    # Override variant at CLI
    python -m landmarking.scripts.evaluate \
        --config landmarking/config/defaults/wflw.json \
        --variant fused_global \
        --checkpoint runs/wflw/fused_global/.../checkpoints/best.pth
"""

import argparse
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
from ..evaluation.metrics_wflw import compute_nme, compute_fr, compute_auc, get_iod_indices_in_subset
from ..training.utils import set_seed, get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ATTR_NAMES = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a trained landmark detection model."
    )
    parser.add_argument("--config", type=str, required=True, help="Config JSON path.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth).")
    parser.add_argument("--split", type=str, default=None, help="Split JSON path (uses test partition). If not set, uses config.dataset.split_path.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path for results.")
    parser.add_argument("--variant", type=str, default=None, help="Override model variant.")
    parser.add_argument("--device", type=str, default=None, help="Override device.")
    parser.add_argument("--mean-shape", type=str, default=None, help="Override mean shape path (GCN models).")
    parser.add_argument(
        "--no-merge", action="store_true",
        help="PIPNet only: use the direct (argmax cell + offset) decode instead "
             "of the neighbor-averaged merge. Useful to isolate whether the NRM "
             "merge helps or hurts on a given checkpoint.",
    )
    return parser.parse_args(argv)


# Variants whose forward signature is (imgs, edge_index) -> (heatmaps, coords),
# i.e. they take a graph but need NO coordinate initialization. These must not be
# dispatched like the GCN/fused family, whose signature is
# (imgs, initial_coords, edge_index).
#
# Keep in sync with TrainingEngine._is_graph_cond_heatmap. Previously this file
# tested `variant == "graph_cond_heatmap"` by exact equality, so graph_prior_fusion
# fell through to the 3-argument call and raised a TypeError at eval time.
GRAPH_COND_HEATMAP_VARIANTS = ("graph_cond_heatmap", "graph_prior_fusion")


def build_model_kwargs(config) -> dict:
    """Construct model kwargs for evaluation, mirroring TrainingEngine.setup().

    Getting this wrong is silent and fatal: a mismatched `heatmap_size` builds a
    different architecture than the checkpoint was trained with. The previous code
    only passed `heatmap_size` when variant == "heatmap", so the graph-conditioned
    heatmap variants were built at the default 64 regardless of config.
    """
    variant = config.model.variant
    kwargs = {"num_landmarks": config.dataset.num_landmarks}

    if variant in GRAPH_COND_HEATMAP_VARIANTS:
        kwargs.update({
            "gnn_hidden": config.model.gnn_hidden,
            "num_layers": config.model.num_layers,
            "heatmap_size": config.model.heatmap_size,
        })
        if variant == "graph_cond_heatmap":
            kwargs["num_heads"] = getattr(config.model, "num_heads", 4)
        if variant == "graph_prior_fusion":
            kwargs.update({
                "sigma_min": config.model.prior_sigma_min,
                "sigma_span": config.model.prior_sigma_span,
                "offset_scale": config.model.prior_offset_scale,
                "chol_bias": getattr(config.model, "prior_chol_bias", 2.0),
                "decode_mode": getattr(config.model, "decode_mode", "windowed"),
                "decode_radius": getattr(config.model, "decode_radius", 5),
                "bn_momentum": getattr(config.model, "bn_momentum", 0.1),
                "prior_disabled": config.model.prior_disabled,
            })
        return kwargs

    if variant == "heatmap":
        kwargs.update({
            "heatmap_size": getattr(config.model, "heatmap_size", 64),
            "decode_mode": getattr(config.model, "decode_mode", "windowed"),
            "decode_radius": getattr(config.model, "decode_radius", 5),
            "bn_momentum": getattr(config.model, "bn_momentum", 0.01),
            # A STAR heatmap checkpoint has sigma_head weights, so the model must
            # be built with use_star to load them.
            "use_star": getattr(config.model, "heatmap_use_star", False),
        })
        return kwargs

    if variant == "hrnet_cascade":
        # Mirror TrainingEngine.setup so the architecture matches the checkpoint
        # (num_stages / shared_weights change the parameter set).
        kwargs.update({
            "num_stages": getattr(config.model, "num_stages", 3),
            "shared_weights": getattr(config.model, "shared_weights", True),
            "heatmap_size": getattr(config.model, "heatmap_size", 128),
            "decode_mode": getattr(config.model, "decode_mode", "windowed"),
            "decode_radius": getattr(config.model, "decode_radius", 5),
            "bn_momentum": getattr(config.model, "bn_momentum", 0.1),
            "cascade_width": getattr(config.model, "cascade_width", 256),
        })
        return kwargs

    if variant == "pipnet":
        # Mirror TrainingEngine.setup: derive neighbor indices from the mean
        # shape so the built architecture matches the checkpoint exactly.
        from ..models.pipnet import get_meanface_indices
        mean_shape = None
        if config.dataset.mean_shape_path and Path(config.dataset.mean_shape_path).exists():
            mean_shape = torch.load(
                config.dataset.mean_shape_path, map_location="cpu", weights_only=False
            )
            if config.dataset.landmark_indices:
                mean_shape = mean_shape[config.dataset.landmark_indices]
        if mean_shape is None:
            raise ValueError(
                "pipnet evaluation requires dataset.mean_shape_path to derive "
                "neighbor indices."
            )
        mf_idx, _, _, _ = get_meanface_indices(mean_shape, config.model.num_nb)
        kwargs.update({
            "backbone": config.model.backbone,
            "pretrained": False,  # weights come from the checkpoint
            "input_size": config.dataset.input_size,
            "net_stride": config.model.net_stride,
            "num_nb": config.model.num_nb,
            "meanface_indices": mf_idx,
            # Must match training: a STAR checkpoint has sigma_layer weights, so
            # the model has to be built with use_star to load them.
            "use_star": getattr(config.model, "pipnet_use_star", False),
        })
        return kwargs

    # GCN / fused / coord family.
    kwargs.update({
        "feat_dim": config.model.feat_dim,
        "gnn_hidden": config.model.gnn_hidden,
        "num_layers": config.model.num_layers,
        "num_iters": config.model.num_iters,
    })
    if variant in ("multiscale", "fused", "fused_global_ms"):
        kwargs["scale_indices"] = config.model.scale_indices
    if variant == "fused_global_patch":
        kwargs.update({
            "patch_mode": config.model.patch_mode,
            "patch_step": config.model.patch_step,
            "patch_radius": config.model.patch_radius,
            "patch_radii": tuple(config.model.patch_radii),
            "patch_proj_dim": config.model.patch_proj_dim,
        })
    if variant == "coord":
        kwargs["use_coarse_init"] = config.model.use_coarse_init
    if variant == "hinit":
        kwargs["heatmap_checkpoint"] = config.model.heatmap_checkpoint
    return kwargs


def evaluate_wflw_gcn(model, test_loader, mean_shape, edge_index, device, num_landmarks, iod_left=60, iod_right=72, variant=None):
    """Evaluate a GCN model on WFLW test set. Returns NME/FR/AUC."""
    model.eval()
    nme_buckets = {name: [] for name in ["full"] + ATTR_NAMES}

    with torch.no_grad():
        for batch in test_loader:
            imgs, coords, metadata = batch
            imgs = imgs.to(device)
            coords = coords.to(device)
            B = imgs.shape[0]

            if variant in GRAPH_COND_HEATMAP_VARIANTS:
                # forward(imgs, edge_index) -> (heatmaps, coords); no init coords.
                _, pred_coords = model(imgs, edge_index)
            else:
                # Mean shape initialization (no noise at eval)
                if mean_shape is not None:
                    initial_coords = mean_shape.unsqueeze(0).expand(B, -1, -1)
                else:
                    initial_coords = torch.full((B, num_landmarks, 2), 0.5, device=device)

                out = model(imgs, initial_coords, edge_index)
                pred_coords = out[0] if isinstance(out, tuple) else out

            # Convert to 512px space for NME
            pred_px = pred_coords.cpu().numpy() * 512.0
            gt_px = coords.cpu().numpy() * 512.0

            for i in range(B):
                nme = compute_nme(pred_px[i], gt_px[i], iod_left=iod_left, iod_right=iod_right)
                if nme is None:
                    continue
                nme_buckets["full"].append(nme)

                # Per-attribute subsets
                if "attrs" in metadata:
                    attrs = metadata["attrs"]
                    if hasattr(attrs, "numpy"):
                        attrs_np = attrs[i].numpy() if attrs.dim() > 1 else attrs.numpy()
                    else:
                        attrs_np = np.array(attrs[i]) if isinstance(attrs, list) else attrs
                    for j, attr_name in enumerate(ATTR_NAMES):
                        if j < len(attrs_np) and attrs_np[j] == 1:
                            nme_buckets[attr_name].append(nme)

    subset_keys = ["full"] + ATTR_NAMES
    results = {
        "nme": {k: float(np.mean(nme_buckets[k])) if nme_buckets[k] else None for k in subset_keys},
        "fr": {k: compute_fr(nme_buckets[k]) for k in subset_keys},
        "auc": {k: compute_auc(nme_buckets[k]) for k in subset_keys},
        "counts": {k: len(nme_buckets[k]) for k in subset_keys},
    }
    return results


def evaluate_wflw_heatmap(model, test_loader, device, heatmap_size, iod_left=None, iod_right=None):
    """Evaluate a heatmap model on WFLW test set using reference decode_preds."""
    from ..evaluation.decode_preds import decode_preds, compute_nme as compute_nme_ref

    model.eval()
    nme_buckets = {name: [] for name in ["full"] + ATTR_NAMES}

    with torch.no_grad():
        for batch in test_loader:
            imgs, target_hm, meta = batch
            imgs = imgs.to(device)
            B = imgs.shape[0]

            pred_hm, _ = model(imgs)
            score_map = pred_hm.cpu()

            preds = decode_preds(
                score_map, meta["center"], meta["scale"],
                [heatmap_size, heatmap_size],
            )

            nme_batch = compute_nme_ref(preds, meta, iod_left=iod_left, iod_right=iod_right)

            # Bucket by full + per-attribute subset
            for i in range(B):
                nme_val = float(nme_batch[i])
                nme_buckets["full"].append(nme_val)

                if "attrs" in meta:
                    attrs = meta["attrs"]
                    if hasattr(attrs, "numpy"):
                        attrs_i = attrs[i].numpy() if attrs.dim() > 1 else attrs.numpy()
                    elif isinstance(attrs, torch.Tensor):
                        attrs_i = attrs[i].numpy()
                    else:
                        attrs_i = np.array(attrs[i]) if isinstance(attrs, list) else None
                    if attrs_i is not None:
                        for j, attr_name in enumerate(ATTR_NAMES):
                            if j < len(attrs_i) and attrs_i[j] == 1:
                                nme_buckets[attr_name].append(nme_val)

    subset_keys = ["full"] + ATTR_NAMES
    results = {
        "nme": {k: float(np.mean(nme_buckets[k])) if nme_buckets[k] else None for k in subset_keys},
        "fr": {k: compute_fr(nme_buckets[k]) for k in subset_keys},
        "auc": {k: compute_auc(nme_buckets[k]) for k in subset_keys},
        "counts": {k: len(nme_buckets[k]) for k in subset_keys},
    }
    return results


def evaluate_cephalometric_split(
    model, loader, device, config, is_heatmap, edge_index, mean_shape,
):
    """Evaluate a cephalometric model on one split's dataloader.

    Runs variant-aware inference (mirroring the TrainingEngine dispatch),
    collects normalized predicted and ground-truth coordinates plus per-sample
    ``orig_size`` / ``pixel_spacing`` metadata, converts to per-landmark radial
    error in millimeters, and returns the MRE/std/SDR summary. Never computes
    the framework NME (Req 7.5).
    """
    from ..evaluation.metrics_cephalometric import (
        compute_radial_error_mm, compute_mre_sdr,
    )

    num_landmarks = config.dataset.num_landmarks
    is_coord_only = config.model.variant in ("hrnet_coord",)
    is_graph_cond_heatmap = config.model.variant in GRAPH_COND_HEATMAP_VARIANTS

    radial_errors_mm = []
    n_samples = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs, coords, *rest = batch
            imgs = imgs.to(device)
            coords = coords.to(device)
            B = imgs.shape[0]
            metadata = rest[-1] if rest else {}

            # Variant-aware forward pass (mirror the TrainingEngine).
            if is_graph_cond_heatmap:
                _, pred = model(imgs, edge_index)
            elif is_heatmap:
                # Heatmap model on coords dataset: forward(imgs) -> (hm, coords)
                _, pred = model(imgs)
            elif is_coord_only:
                pred = model(imgs)
            else:
                # GCN / fused variants: forward(imgs, init, edge_index)
                if mean_shape is not None:
                    init = mean_shape.unsqueeze(0).expand(B, -1, -1)
                else:
                    init = torch.full(
                        (B, num_landmarks, 2), 0.5, device=device
                    )
                out = model(imgs, init, edge_index)
                pred = out[0] if isinstance(out, tuple) else out

            pred_norm = pred.cpu().numpy()   # (B, K, 2) in [0,1]
            gt_norm = coords.cpu().numpy()   # (B, K, 2) in [0,1]

            # Per-sample metadata (batched tensors -> index per sample).
            orig_size = metadata.get("orig_size") if hasattr(metadata, "get") else None
            pixel_spacing = metadata.get("pixel_spacing") if hasattr(metadata, "get") else None

            for i in range(B):
                # orig_size[i] -> [H, W]
                if orig_size is not None:
                    os_i = orig_size[i]
                    os_i = os_i.tolist() if hasattr(os_i, "tolist") else list(os_i)
                else:
                    os_i = [config.dataset.input_size, config.dataset.input_size]

                # pixel_spacing[i] -> scalar
                if pixel_spacing is not None:
                    ps_i = pixel_spacing[i]
                    ps_i = float(ps_i.item()) if hasattr(ps_i, "item") else float(ps_i)
                else:
                    ps_i = float(config.dataset.pixel_spacing)

                errs = compute_radial_error_mm(
                    pred_norm[i], gt_norm[i], os_i, ps_i
                )
                radial_errors_mm.extend([float(e) for e in errs])
                n_samples += 1

    summary = compute_mre_sdr(radial_errors_mm)
    summary["n_samples"] = n_samples
    return summary


def main(argv=None):
    args = parse_args(argv)

    # Load config
    config = LandmarkingConfig.from_json(args.config)
    config.resolve_paths()

    if args.variant:
        config.model.variant = args.variant
    if args.device:
        config.training.device = args.device
    if args.mean_shape:
        config.dataset.mean_shape_path = args.mean_shape

    device = get_device(config.training.device)
    set_seed(config.training.seed)
    is_heatmap = config.model.variant == "heatmap"

    # Load split
    split_path = args.split or config.dataset.split_path
    # For cephalometric, evaluation reports Test1 and Test2 separately. We
    # collect a {split_name: [paths]} mapping so each partition is scored on
    # its own. Precedence matches other datasets:
    #   --split -> config.dataset.split_path -> directory discovery.
    ceph_splits = None
    if split_path and Path(split_path).exists():
        with open(split_path) as f:
            split_data = json.load(f)
        test_paths = split_data.get("test", [])
        if config.dataset.name == "cephalometric":
            # A provided split file is honored as a single "test" list.
            ceph_splits = {"test": test_paths}
    elif config.dataset.name == "cephalometric":
        # Cephalometric: no split file -> discover Test1 and Test2 separately
        # so each partition is scored on its own (Req 7.6, 10.3, 11.5).
        data_dir = Path(config.dataset.data_dir)
        ceph_splits = {}
        for split_name in ("test1", "test2"):
            split_dir = data_dir / split_name
            if split_dir.exists():
                paths = sorted([str(p) for p in split_dir.glob("*.pt")])
                if paths:
                    ceph_splits[split_name] = paths
                    logger.info(
                        f"Auto-discovered {split_name} set "
                        f"({len(paths)} samples) from: {split_dir}"
                    )
        if not ceph_splits:
            logger.error(
                f"No split file and no test1/test2 directories with .pt "
                f"files found under {data_dir}. Provide --split or ensure "
                f"test data exists."
            )
            sys.exit(1)
        # test_paths retained for the shared logging line below.
        test_paths = [p for paths in ceph_splits.values() for p in paths]
    else:
        # Auto-discover test set from data directory
        data_dir = Path(config.dataset.data_dir)
        if config.dataset.name == "wflw":
            # WFLW always uses the official test set from pt_crops/test/
            test_dir = data_dir / "pt_crops" / "test"
        else:
            test_dir = data_dir / "test"

        if test_dir.exists():
            test_paths = sorted([str(p) for p in test_dir.glob("*.pt")])
            logger.info(f"Auto-discovered test set from: {test_dir}")
        else:
            logger.error(
                f"No split file and no test directory found at {test_dir}. "
                f"Provide --split or ensure test data exists."
            )
            sys.exit(1)

    if not test_paths:
        logger.error("No test files found.")
        sys.exit(1)
    logger.info(f"Test set: {len(test_paths)} samples")

    # Build model (kwargs mirror TrainingEngine.setup so the architecture matches
    # the checkpoint exactly).
    model_kwargs = build_model_kwargs(config)
    model = get_model(config.model.variant, **model_kwargs)

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # Build test dataloader
    if is_heatmap and config.dataset.name == "wflw":
        from ..datasets.wflw.dataset_ref import WFLWRefDataset
        test_ds = WFLWRefDataset(
            pt_paths=test_paths,
            augment=False,
            landmark_indices=config.dataset.landmark_indices or None,
        )
    elif config.dataset.name == "wflw":
        from ..datasets.wflw.dataset import WFLWDataset
        test_ds = WFLWDataset(
            pt_paths=test_paths,
            input_size=config.dataset.input_size,
            num_landmarks=config.dataset.num_landmarks,
            augment=False,
            landmark_indices=config.dataset.landmark_indices or None,
        )
    elif config.dataset.name == "cephalometric":
        from ..datasets.cephalometric.dataset import CephalometricDataset

        ceph_mode = "heatmap" if is_heatmap else "coord"

        def _build_ceph_loader(paths, split_name):
            ds = CephalometricDataset(
                pt_paths=paths,
                input_size=config.dataset.input_size,
                num_landmarks=config.dataset.num_landmarks,
                augment=False,
                mode=ceph_mode,
                heatmap_size=config.model.heatmap_size,
                sigma=config.model.sigma,
                pixel_spacing=config.dataset.pixel_spacing,
                landmark_indices=config.dataset.landmark_indices or None,
                split=split_name,
            )
            return DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

        # Per-split loaders are built inside the cephalometric eval branch;
        # test_loader is unused for this dataset.
        test_loader = None
    else:
        from ..datasets.lizard.dataset import LizardDataset
        test_ds = LizardDataset(
            pt_paths=test_paths,
            input_size=config.dataset.input_size,
            num_landmarks=config.dataset.num_landmarks,
            augment=False,
        )

    if config.dataset.name != "cephalometric":
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    # Run evaluation
    if is_heatmap and config.dataset.name == "wflw":
        # Compute IOD positions for subset
        if config.dataset.landmark_indices:
            iod_left, iod_right = get_iod_indices_in_subset(config.dataset.landmark_indices)
        else:
            iod_left, iod_right = None, None
        results = evaluate_wflw_heatmap(model, test_loader, device, config.model.heatmap_size, iod_left=iod_left, iod_right=iod_right)
    elif config.dataset.name == "wflw":
        # Determine IOD positions (handle subsets)
        if config.dataset.landmark_indices:
            iod_left, iod_right = get_iod_indices_in_subset(config.dataset.landmark_indices)
        else:
            iod_left, iod_right = 60, 72

        # Load mean shape for GCN
        mean_shape = None
        if config.dataset.mean_shape_path and Path(config.dataset.mean_shape_path).exists():
            mean_shape = torch.load(config.dataset.mean_shape_path, map_location=device, weights_only=False)
            # Subsample mean shape for subsets
            if config.dataset.landmark_indices:
                mean_shape = mean_shape[config.dataset.landmark_indices]
        edge_index = get_edge_index(config.dataset.graph_topology, config.dataset.num_landmarks, landmark_indices=config.dataset.landmark_indices or None).to(device)
        results = evaluate_wflw_gcn(model, test_loader, mean_shape, edge_index, device, config.dataset.num_landmarks, iod_left=iod_left, iod_right=iod_right, variant=config.model.variant)
    elif config.dataset.name == "cephalometric":
        # Cephalometric evaluation — MRE (mm) + SDR, reported per split
        # (Test1 / Test2 separately). No framework NME (Req 7.5).
        edge_index = get_edge_index(
            config.dataset.graph_topology, config.dataset.num_landmarks,
            landmark_indices=config.dataset.landmark_indices or None,
        ).to(device)
        mean_shape = None
        if config.dataset.mean_shape_path and Path(config.dataset.mean_shape_path).exists():
            mean_shape = torch.load(
                config.dataset.mean_shape_path, map_location=device, weights_only=False
            )
            if config.dataset.landmark_indices:
                mean_shape = mean_shape[config.dataset.landmark_indices]

        results = {}
        for split_name, split_paths in ceph_splits.items():
            loader = _build_ceph_loader(split_paths, split_name)
            results[split_name] = evaluate_cephalometric_split(
                model, loader, device, config, is_heatmap, edge_index, mean_shape,
            )
    else:
        # Lizard evaluation — pixel error
        from ..evaluation.metrics_lizard import compute_pixel_error, pixel_to_mm
        model.eval()
        errors = []
        mm_errors = []
        edge_index = get_edge_index(config.dataset.graph_topology, config.dataset.num_landmarks).to(device)
        mean_shape = None
        if config.dataset.mean_shape_path and Path(config.dataset.mean_shape_path).exists():
            mean_shape = torch.load(config.dataset.mean_shape_path, map_location=device, weights_only=False)

        # PIPNet: give the model its reverse-neighbor index so the merged
        # (neighbor-averaged) decode used at inference is available.
        is_pipnet = config.model.variant == "pipnet"
        pip_merge = is_pipnet and not getattr(args, "no_merge", False)
        if is_pipnet:
            from ..models.pipnet import get_meanface_indices
            ms_cpu = mean_shape
            if ms_cpu is None:
                raise ValueError("pipnet evaluation requires a mean_shape_path.")
            _, rev1, rev2, rev_ml = get_meanface_indices(ms_cpu.cpu(), config.model.num_nb)
            model.set_reverse_index(rev1, rev2, rev_ml)
            logger.info(
                f"PIPNet decode: {'neighbor-merged' if pip_merge else 'direct (no merge)'}"
            )

        with torch.no_grad():
            for batch in test_loader:
                imgs, coords, *rest = batch
                imgs = imgs.to(device)
                coords = coords.to(device)
                B = imgs.shape[0]
                if is_pipnet:
                    # Direct or neighbor-averaged decode per --no-merge.
                    pred = model.predict_coords(imgs, merge=pip_merge)
                elif config.model.variant == "heatmap":
                    # HRNet heatmap: forward(imgs) -> (heatmaps, coords). Works
                    # for both plain and STAR variants (forward is unchanged;
                    # STAR only adds forward_star used in training). Take coords.
                    _, pred = model(imgs)
                elif config.model.variant == "hrnet_cascade":
                    # forward(imgs) -> (list[stage heatmaps], final coords).
                    _, pred = model(imgs)
                elif config.model.variant in GRAPH_COND_HEATMAP_VARIANTS:
                    # forward(imgs, edge_index) -> (heatmaps, coords); no init.
                    _, pred = model(imgs, edge_index)
                else:
                    if mean_shape is not None:
                        init = mean_shape.unsqueeze(0).expand(B, -1, -1)
                    else:
                        init = torch.full((B, config.dataset.num_landmarks, 2), 0.5, device=device)
                    out = model(imgs, init, edge_index)
                    pred = out[0] if isinstance(out, tuple) else out
                pred_px = pred.cpu().numpy() * config.dataset.input_size
                gt_px = coords.cpu().numpy() * config.dataset.input_size

                # Extract metadata (ruler_px for mm conversion)
                metadata = rest[0] if rest else {}

                for i in range(B):
                    px_err = compute_pixel_error(pred_px[i], gt_px[i])
                    errors.append(px_err.mean())

                    # Convert to mm if ruler_px is available
                    if "ruler_px" in metadata:
                        ruler_val = float(metadata["ruler_px"][i])
                        if ruler_val > 0:
                            mm_err = pixel_to_mm(px_err, ruler_val, ruler_mm=10.0)
                            mm_errors.append(mm_err.mean())

        results = {
            "mean_px_error": float(np.mean(errors)),
            "median_px_error": float(np.median(errors)),
            "n_evaluated": len(errors),
        }
        if is_pipnet:
            results["pipnet_decode"] = "merged" if pip_merge else "direct"
        if mm_errors:
            results["mean_mm_error"] = float(np.mean(mm_errors))
            results["median_mm_error"] = float(np.median(mm_errors))
            results["n_mm_samples"] = len(mm_errors)

    # Log results
    logger.info("=" * 50)
    logger.info(f"Results ({config.dataset.name} / {config.model.variant}):")
    if "nme" in results:
        for k in results["nme"]:
            nme_v = results["nme"][k]
            fr_v = results["fr"][k]
            auc_v = results["auc"][k]
            n = results["counts"][k]
            nme_s = f"{nme_v:.4f}" if nme_v is not None else "N/A"
            logger.info(f"  {k:<16} NME={nme_s}  FR={fr_v:.4f}  AUC={auc_v:.4f}  (n={n})")
    elif config.dataset.name == "cephalometric":
        # Per-split MRE (mm) + std + SDR@2/2.5/3/4mm
        for split_name, split_res in results.items():
            mre_v = split_res.get("mre")
            std_v = split_res.get("std")
            n = split_res.get("n", 0)
            mre_s = f"{mre_v:.4f}" if mre_v is not None else "N/A"
            std_s = f"{std_v:.4f}" if std_v is not None else "N/A"
            logger.info(
                f"  [{split_name}] MRE={mre_s} mm  std={std_s} mm  "
                f"(n_landmarks={n}, n_samples={split_res.get('n_samples', 0)})"
            )
            sdr = split_res.get("sdr", {})
            for thr in ("2.0mm", "2.5mm", "3.0mm", "4.0mm"):
                sdr_v = sdr.get(thr)
                sdr_s = f"{sdr_v:.2f}%" if sdr_v is not None else "N/A"
                logger.info(f"      SDR@{thr:<6} = {sdr_s}")
    else:
        logger.info(f"  Mean pixel error: {results['mean_px_error']:.2f}")
        logger.info(f"  Median pixel error: {results['median_px_error']:.2f}")
        if "mean_mm_error" in results:
            logger.info(f"  Mean mm error: {results['mean_mm_error']:.3f}")
            logger.info(f"  Median mm error: {results['median_mm_error']:.3f}")
            logger.info(f"  Samples with ruler: {results['n_mm_samples']}")
        logger.info(f"  Samples evaluated: {results['n_evaluated']}")
    logger.info("=" * 50)

    # Save results
    output_path = args.output
    if not output_path:
        output_path = str(Path(args.checkpoint).parent.parent / "eval_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
