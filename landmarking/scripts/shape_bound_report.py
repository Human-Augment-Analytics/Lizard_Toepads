"""Oracle shape-space reconstruction bound: can a shape-coefficient model win?

Fits a point-distribution model (Procrustes + PCA) on the TRAIN split and
reports, for each component count k, the lowest per-landmark error achievable
on HELD-OUT shapes by any model that outputs a similarity transform plus k
shape coefficients. Both the transform and the coefficients are fitted to the
ground truth, so the numbers are a strict lower bound even for a perfect image
encoder.

Use this to decide whether to build such a model at all. If the bound at your
intended k already exceeds the error of an existing unconstrained model, a
k-component shape model cannot beat it and the basis needs more components or
must be nonlinear.

Context for reading the output: HRNet-heatmap on Lizard emits
``num_landmarks * heatmap_size^2`` values (147,456 at N=9, 128px) to describe
an object with at most ``2N - 4`` = 14 shape degrees of freedom. The bound
tells you what that 14-DOF description actually costs in pixels.

Expect a smooth decay in k, not a cliff. A cliff only appears when shape
variation is confined to a linear subspace orthogonal to the similarity group,
which real morphometric data is not.

Usage:
    # Cluster, via the experiment framework (resolves data_dir / split_path):
    python -m landmarking.scripts.shape_bound_report \
        --config landmarking/config/defaults/lizard.json \
        --baseline-px 6.5

    # Split finger and toe into separate shape models:
    python -m landmarking.scripts.shape_bound_report \
        --config landmarking/config/defaults/lizard.json \
        --by-class --baseline-px 6.5

    # Explicit directory instead of a config:
    python -m landmarking.scripts.shape_bound_report \
        --data-dir /home/hice1/axu39/scratch/Lizard_data/lizard/train

    # Local dry run against consolidated TPS files (no .pt, no cluster):
    python -m landmarking.scripts.shape_bound_report \
        --tps ml-morph/consolidated_finger.tps --canvas 0
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

from ..common.shape_space import centroid_size, fit_shape_model, reconstruction_bound

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Component counts are reported against these variance targets so the table
# answers "how many modes do I need" without the reader doing arithmetic.
VARIANCE_TARGETS = (0.90, 0.95, 0.99)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _to_float(value, default=None):
    """Coerce a tensor/scalar/None to float, mirroring lizard_resolution_report."""
    if value is None:
        return default
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        return default


def discover_pt_paths(config, seed: int = 42) -> tuple:
    """Resolve train/val .pt paths exactly as TrainingEngine does.

    Mirrors ``TrainingEngine._create_dataloaders``: prefer an explicit
    ``dataset.split_path``, otherwise auto-discover from
    ``data_dir/pt_crops/train`` (WFLW layout) or ``data_dir/train`` (Lizard
    layout) and take a deterministic 80/20 split. Matching the engine matters
    because the bound must be computed on the same split the models trained on;
    a different split would make the comparison against reported model error
    invalid.

    Args:
        config: A resolved :class:`LandmarkingConfig`.
        seed: Seed for the auto-split, should match ``training.seed``.

    Returns:
        Tuple ``(train_paths, val_paths)`` of string lists.

    Raises:
        FileNotFoundError: If no data directory or no .pt files are found.
    """
    split_path = config.dataset.split_path
    if split_path and Path(split_path).exists():
        with open(split_path) as f:
            split_data = json.load(f)
        logger.info(f"Using split file: {split_path}")
        return split_data.get("train", []), split_data.get("val", [])

    data_dir = Path(config.dataset.data_dir)
    candidates = [data_dir / "pt_crops" / "train", data_dir / "train", data_dir]
    pt_dir = next((c for c in candidates if c.exists()), None)
    if pt_dir is None:
        raise FileNotFoundError(
            f"No data directory found. Searched: {[str(c) for c in candidates]}"
        )

    all_paths = sorted(str(p) for p in pt_dir.glob("*.pt"))
    if not all_paths:
        raise FileNotFoundError(f"No .pt files found in {pt_dir}")

    rng = random.Random(seed)
    shuffled = list(all_paths)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * 0.8)
    logger.info(
        f"Auto-split from {pt_dir}: {cut} train, {len(shuffled) - cut} val "
        f"(seed={seed}, matches TrainingEngine)"
    )
    return sorted(shuffled[:cut]), sorted(shuffled[cut:])


def load_pt_shapes(paths, num_landmarks: int) -> dict:
    """Load 'tps' landmark arrays and per-sample metadata from .pt crops.

    Args:
        paths: Iterable of .pt file paths.
        num_landmarks: Expected landmark count; mismatches are skipped.

    Returns:
        Dict with ``shapes`` (S, N, 2) float64 array in canvas pixels,
        ``ruler_px`` list, ``class_name`` list, and ``skipped`` count.
    """
    import torch

    shapes, rulers, classes, skipped = [], [], [], 0
    for path in paths:
        try:
            data = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.warning(f"Failed to load {path}: {exc}")
            skipped += 1
            continue

        tps = data.get("tps")
        if tps is None:
            skipped += 1
            continue
        arr = np.asarray(tps, dtype=np.float64)
        if arr.shape != (num_landmarks, 2):
            skipped += 1
            continue

        shapes.append(arr)
        rulers.append(_to_float(data.get("ruler_px")))
        cls = data.get("class_name")
        classes.append(str(cls) if cls is not None else None)

    return {
        "shapes": np.stack(shapes) if shapes else np.zeros((0, num_landmarks, 2)),
        "ruler_px": rulers,
        "class_name": classes,
        "skipped": skipped,
    }


def load_tps_shapes(paths, num_landmarks: int) -> dict:
    """Load shapes from consolidated TPS files (local dry-run path).

    Args:
        paths: Iterable of consolidated .tps file paths.
        num_landmarks: Expected anatomical landmark count.

    Returns:
        Same structure as :func:`load_pt_shapes`. ``class_name`` is inferred
        from the filename when it contains "finger" or "toe".
    """
    from ..common.tps_io import read_consolidated_tps

    shapes, rulers, classes = [], [], []
    for path in paths:
        stem = Path(path).stem.lower()
        cls = "finger" if "finger" in stem else ("toe" if "toe" in stem else None)
        for spec in read_consolidated_tps(str(path), expected_landmarks=num_landmarks):
            shapes.append(spec["landmarks"])
            rulers.append(spec["ruler_px"])
            classes.append(cls)

    return {
        "shapes": np.stack(shapes) if shapes else np.zeros((0, num_landmarks, 2)),
        "ruler_px": rulers,
        "class_name": classes,
        "skipped": 0,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _k_for_variance(model, target: float) -> int:
    """Smallest k whose cumulative explained-variance ratio reaches ``target``."""
    cum = model.cumulative_variance_ratio
    hits = np.nonzero(cum >= target)[0]
    return int(hits[0]) + 1 if hits.size else int(model.n_components)


def _median_mm_per_px(ruler_px, ruler_mm: float):
    """Median mm-per-pixel from available ruler measurements, or None."""
    valid = [r for r in ruler_px if r is not None and r > 0]
    if not valid:
        return None
    return float(ruler_mm / np.median(valid))


def rescale_long_side(shapes: np.ndarray, target: float) -> np.ndarray:
    """Scale each shape about its centroid so its bounding-box long side equals
    ``target``.

    Needed to make raw-TPS results comparable to model error. Consolidated TPS
    files store coordinates in original high-resolution image space, where a
    toepad spans ~600-800 px, whereas the trained models see OBB-cropped,
    letterboxed 512 canvases where it spans ~512. Errors in the two frames
    differ by that ratio, so comparing a TPS-derived bound against a
    canvas-pixel baseline without rescaling is meaningless.

    This is exact rather than approximate in the sense that matters: the bound
    is similarity-equivariant (scaling every shape by c scales the error by c),
    so rescaling before analysis is equivalent to rescaling the reported error.
    It only *approximates* the real preprocessing, which letterboxes the image
    rather than the landmark hull.

    Args:
        shapes: (S, N, 2) shapes.
        target: Desired bounding-box long side, in the output units.

    Returns:
        (S, N, 2) rescaled shapes.

    Raises:
        ValueError: If ``target`` is not positive.
    """
    if target <= 0:
        raise ValueError(f"target must be positive, got {target}")
    arr = np.asarray(shapes, dtype=np.float64)
    out = np.empty_like(arr)
    for i, shape in enumerate(arr):
        centroid = shape.mean(axis=0)
        extent = shape.max(axis=0) - shape.min(axis=0)
        long_side = float(extent.max())
        factor = target / long_side if long_side > 0 else 1.0
        out[i] = (shape - centroid) * factor + centroid
    return out


def analyze(
    train_shapes: np.ndarray,
    val_shapes: np.ndarray,
    allow_rotation: bool,
    baseline_px=None,
    mm_per_px=None,
) -> dict:
    """Fit a shape model on train shapes and bound reconstruction on val shapes.

    Args:
        train_shapes: (S_train, N, 2) shapes used to fit the basis.
        val_shapes: (S_val, N, 2) held-out shapes used for the bound.
        allow_rotation: Whether rotation is factored out as nuisance. Set False
            when an upstream OBB crop already canonicalized orientation and you
            want residual rotation counted as error the model must predict.
        baseline_px: Optional existing model error to compare against.
        mm_per_px: Optional scalar for reporting the bound in millimetres.

    Returns:
        A JSON-serializable dict describing the model and the bound.
    """
    model = fit_shape_model(train_shapes, allow_rotation=allow_rotation)
    bound = reconstruction_bound(val_shapes, model)

    # Median centroid size of the held-out shapes, so the bound can also be
    # expressed as a percentage of object size. That form is unit-free and stays
    # comparable across datasets, canvases, and coordinate frames.
    val_sizes = [centroid_size(s) for s in val_shapes]
    median_cs = float(np.median(val_sizes)) if val_sizes else None

    cum = model.cumulative_variance_ratio
    result = {
        "allow_rotation": allow_rotation,
        "n_train": int(train_shapes.shape[0]),
        "n_val": int(val_shapes.shape[0]),
        "num_landmarks": model.num_landmarks,
        "effective_rank": model.effective_rank,
        "n_components": model.n_components,
        "explained_variance_ratio": [float(v) for v in model.explained_variance_ratio],
        "cumulative_variance_ratio": [float(v) for v in cum],
        "k_for_variance": {
            f"{t:.2f}": _k_for_variance(model, t) for t in VARIANCE_TARGETS
        },
        "mean_shape": model.mean.tolist(),
        "bound": bound,
        "mm_per_px": mm_per_px,
        "baseline_px": baseline_px,
        "median_centroid_size": median_cs,
    }

    # Per-shape means give a standard error, which is what tells you whether a
    # gap between the bound and a model's error is larger than sampling noise.
    stderr = []
    for per_shape in bound["per_shape_mean"]:
        arr = np.asarray(per_shape, dtype=np.float64)
        stderr.append(
            float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
        )
    result["bound"]["stderr_of_mean"] = stderr

    if baseline_px is not None:
        feasible = [
            k
            for k, err in zip(bound["k_values"], bound["mean_error"])
            if err < baseline_px
        ]
        result["min_k_beating_baseline"] = int(feasible[0]) if feasible else None

    return result


def print_report(name: str, res: dict) -> None:
    """Print one analysis block as a human-readable table."""
    rot = "similarity (rotation factored out)" if res["allow_rotation"] else (
        "translation + scale only (rotation counted as error)"
    )
    print()
    print("=" * 78)
    print(f"{name}  |  {rot}")
    print("=" * 78)
    print(
        f"train={res['n_train']}  val={res['n_val']}  "
        f"N={res['num_landmarks']}  effective shape DOF={res['effective_rank']}"
    )
    mmpp = res.get("mm_per_px")
    if mmpp:
        print(f"median scale: {mmpp:.5g} mm per canvas pixel")

    print("\nComponents needed to explain variance:")
    for target, k in res["k_for_variance"].items():
        print(f"  {float(target) * 100:.0f}% -> k={k}")

    median_cs = res.get("median_centroid_size")
    print("\nOracle held-out reconstruction bound (per-landmark error):")
    header = f"  {'k':>3} {'cum.var':>8} {'mean px':>9} {'+/-SE':>7} {'median':>8} {'p90':>8}"
    if median_cs:
        header += f" {'%obj':>7}"
    if mmpp:
        header += f" {'mean mm':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    bound = res["bound"]
    cum = res["cumulative_variance_ratio"]
    for i, k in enumerate(bound["k_values"]):
        cv = cum[k - 1] if k > 0 else 0.0
        row = (
            f"  {k:>3} {cv:>8.4f} {bound['mean_error'][i]:>9.3f} "
            f"{bound['stderr_of_mean'][i]:>7.3f} "
            f"{bound['median_error'][i]:>8.3f} {bound['p90_error'][i]:>8.3f}"
        )
        if median_cs:
            row += f" {100.0 * bound['mean_error'][i] / median_cs:>7.3f}"
        if mmpp:
            row += f" {bound['mean_error'][i] * mmpp:>9.4f}"
        print(row)
    if median_cs:
        print(
            f"  %obj = error as a percentage of median centroid size "
            f"({median_cs:.1f}); unit-free and frame-independent."
        )

    baseline = res.get("baseline_px")
    if baseline is not None:
        min_k = res.get("min_k_beating_baseline")
        print(f"\nBaseline to beat: {baseline:.3f} px")
        if min_k is None:
            print(
                "  NO k reaches it. A linear shape model with a perfect encoder "
                "cannot match this baseline -> the basis is the binding "
                "constraint, not the image features."
            )
        else:
            print(
                f"  k={min_k} is the smallest component count whose ORACLE bound "
                f"beats it.\n  A real model must predict the transform and the "
                f"coefficients, so treat k={min_k} as the floor and budget "
                f"headroom above it."
            )

    # Per-landmark breakdown at the most informative k.
    idx = len(bound["k_values"]) - 1
    if baseline is not None and res.get("min_k_beating_baseline") is not None:
        want = res["min_k_beating_baseline"]
        if want in bound["k_values"]:
            idx = bound["k_values"].index(want)
    k_shown = bound["k_values"][idx]
    per_lm = bound["per_landmark_mean"][idx]
    print(f"\nPer-landmark mean error at k={k_shown} (px):")
    print("  " + "".join(f"{i:>8}" for i in range(len(per_lm))))
    print("  " + "".join(f"{v:>8.2f}" for v in per_lm))
    worst = int(np.argmax(per_lm))
    print(
        f"  worst: lm{worst} at {per_lm[worst]:.2f} px "
        f"({per_lm[worst] / max(np.mean(per_lm), 1e-9):.2f}x the mean)"
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Oracle shape-space reconstruction bound for landmark data."
    )
    src = ap.add_argument_group("data source (choose one)")
    src.add_argument("--config", type=str, help="Config JSON; resolves data_dir/split.")
    src.add_argument("--data-dir", type=str, help="Directory of .pt files.")
    src.add_argument("--tps", type=str, nargs="+", help="Consolidated .tps file(s).")

    ap.add_argument("--num-landmarks", type=int, default=None,
                    help="Override landmark count (default: config, else 9).")
    ap.add_argument("--val-fraction", type=float, default=0.2,
                    help="Held-out fraction when no split file exists (default 0.2).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Split seed (default: config training.seed, else 42).")
    ap.add_argument("--baseline-px", type=float, default=None,
                    help="Existing model error in px to compare the bound against.")
    ap.add_argument("--ruler-mm", type=float, default=10.0,
                    help="Physical ruler length in mm (default 10).")
    ap.add_argument("--canvas", type=int, default=None,
                    help="Canvas size for context only; 0 to suppress.")
    ap.add_argument("--rescale-long-side", type=float, default=0.0,
                    help="Rescale each shape so its bounding-box long side equals "
                         "this, making raw-TPS errors comparable to canvas-pixel "
                         "model error. Use 512 with --tps; leave 0 for .pt input, "
                         "which is already in canvas coordinates.")
    ap.add_argument("--by-class", action="store_true",
                    help="Fit separate shape models per class_name (finger/toe).")
    ap.add_argument("--no-rotation-variant", action="store_true",
                    help="Also report the translation+scale-only bound.")
    ap.add_argument("--output", type=str, default=None,
                    help="Write JSON here (default: <output_root>/<name>/shape_bound/).")
    args = ap.parse_args(argv)

    if not any([args.config, args.data_dir, args.tps]):
        ap.error("one of --config, --data-dir, or --tps is required")

    config = None
    num_landmarks = args.num_landmarks
    seed = args.seed
    output_path = args.output

    if args.config:
        from ..config.schema import LandmarkingConfig

        config = LandmarkingConfig.from_json(args.config).resolve_paths()
        config.validate()
        num_landmarks = num_landmarks or config.dataset.num_landmarks
        seed = seed if seed is not None else config.training.seed
        if args.canvas is None:
            args.canvas = config.dataset.input_size
    num_landmarks = num_landmarks or 9
    seed = 42 if seed is None else seed

    # ---- Load ----
    if args.tps:
        data = load_tps_shapes(args.tps, num_landmarks)
        source = ", ".join(args.tps)
        train_pool, val_pool = None, None
    elif args.data_dir:
        paths = sorted(str(p) for p in Path(args.data_dir).glob("*.pt"))
        if not paths:
            print(f"ERROR: no .pt files in {args.data_dir}", file=sys.stderr)
            return 1
        data = load_pt_shapes(paths, num_landmarks)
        source = args.data_dir
        train_pool, val_pool = None, None
    else:
        try:
            train_paths, val_paths = discover_pt_paths(config, seed=seed)
        except FileNotFoundError as exc:
            # Expected whenever the config points at cluster paths that are not
            # mounted. Report it as a message rather than a traceback, and name
            # the escape hatches.
            print(f"ERROR: {exc}", file=sys.stderr)
            print(
                "Run this where the data lives, or point at it directly with "
                "--data-dir, or use --tps for a local dry run.",
                file=sys.stderr,
            )
            return 1
        train_pool = load_pt_shapes(train_paths, num_landmarks)
        val_pool = load_pt_shapes(val_paths, num_landmarks)
        data = None
        source = config.dataset.data_dir

    # When the source was a flat pool, carve out a held-out set here. Fitting and
    # evaluating on the same shapes would report in-sample reconstruction error,
    # which falls toward zero as k grows whether or not the basis generalizes.
    if data is not None:
        n_total = data["shapes"].shape[0]
        if n_total < 10:
            print(f"ERROR: only {n_total} usable shapes; need >= 10", file=sys.stderr)
            return 1
        order = list(range(n_total))
        random.Random(seed).shuffle(order)
        cut = int(n_total * (1.0 - args.val_fraction))
        tr_idx, va_idx = order[:cut], order[cut:]

        def _subset(idx):
            return {
                "shapes": data["shapes"][idx],
                "ruler_px": [data["ruler_px"][i] for i in idx],
                "class_name": [data["class_name"][i] for i in idx],
            }

        train_pool, val_pool = _subset(tr_idx), _subset(va_idx)
        logger.info(
            f"Held-out split from {n_total} shapes: "
            f"{len(tr_idx)} train, {len(va_idx)} val (seed={seed})"
        )

    if train_pool["shapes"].shape[0] < 2:
        print("ERROR: need >= 2 training shapes", file=sys.stderr)
        return 1
    if val_pool["shapes"].shape[0] < 1:
        print("ERROR: held-out set is empty", file=sys.stderr)
        return 1

    if args.rescale_long_side > 0:
        for pool in (train_pool, val_pool):
            pool["shapes"] = rescale_long_side(
                pool["shapes"], args.rescale_long_side
            )
        # Ruler distances were measured in the original frame, so they no longer
        # correspond to the rescaled coordinates. Drop them rather than report mm
        # that are silently wrong by the rescale factor.
        if any(r is not None for r in train_pool["ruler_px"]):
            logger.warning(
                "--rescale-long-side set: discarding ruler_px, so mm output is "
                "suppressed (ruler was measured in the original frame)."
            )
        for pool in (train_pool, val_pool):
            pool["ruler_px"] = [None] * len(pool["ruler_px"])
        logger.info(
            f"Rescaled every shape to a bounding-box long side of "
            f"{args.rescale_long_side:g}."
        )

    print(f"Source: {source}")
    print(f"Landmarks: {num_landmarks}   canvas: {args.canvas or 'n/a'}")

    # ---- Group ----
    groups = {"all": (train_pool, val_pool)}
    if args.by_class:
        present = sorted(
            {c for c in train_pool["class_name"] if c}
            & {c for c in val_pool["class_name"] if c}
        )
        if not present:
            logger.warning("--by-class requested but no class_name present; skipping.")
        for cls in present:
            def _filt(pool, want=cls):
                idx = [i for i, c in enumerate(pool["class_name"]) if c == want]
                return {
                    "shapes": pool["shapes"][idx],
                    "ruler_px": [pool["ruler_px"][i] for i in idx],
                    "class_name": [pool["class_name"][i] for i in idx],
                }

            tr, va = _filt(train_pool), _filt(val_pool)
            if tr["shapes"].shape[0] >= 2 and va["shapes"].shape[0] >= 1:
                groups[cls] = (tr, va)

    # ---- Analyze ----
    rotation_modes = [True] + ([False] if args.no_rotation_variant else [])
    results = {}
    for gname, (tr, va) in groups.items():
        mm_per_px = _median_mm_per_px(tr["ruler_px"], args.ruler_mm)
        results[gname] = {}
        for allow_rotation in rotation_modes:
            res = analyze(
                tr["shapes"],
                va["shapes"],
                allow_rotation=allow_rotation,
                baseline_px=args.baseline_px,
                mm_per_px=mm_per_px,
            )
            key = "similarity" if allow_rotation else "no_rotation"
            results[gname][key] = res
            print_report(f"group={gname}", res)

    # ---- Interpretation ----
    print()
    print("=" * 78)
    print("Interpretation")
    print("=" * 78)
    print("- The bound is an ORACLE: the similarity transform and the shape")
    print("  coefficients are both fitted to the ground-truth shape. A trained")
    print("  model must predict them, so its error is strictly worse.")
    print("- Expect a smooth decay in k, not a cliff. A cliff only occurs when")
    print("  shape variation lies in a linear subspace orthogonal to the")
    print("  similarity group, which real morphometric data does not.")
    print("- If the bound at a small k sits far below your current model error,")
    print("  the shape basis is NOT the constraint and a shape-coefficient model")
    print("  has real headroom. If no k reaches your baseline, a linear basis")
    print("  caps you and you need more components or a nonlinear shape model.")
    print("- Compare gaps against the +/-SE column. A bound that beats a model")
    print("  by less than a couple of standard errors is not a real difference.")
    print("- A landmark whose per-landmark error stays high at large k is poorly")
    print("  predicted by the others' geometry: either genuinely independent")
    print("  variation or annotation ambiguity. No shape prior will fix it.")

    # ---- Persist ----
    if output_path is None and config is not None:
        output_path = str(
            Path(config.paths.output_root)
            / config.dataset.name
            / "shape_bound"
            / "shape_bound.json"
        )
    if output_path:
        payload = {
            "source": source,
            "num_landmarks": num_landmarks,
            "canvas": args.canvas,
            "seed": seed,
            "ruler_mm": args.ruler_mm,
            "baseline_px": args.baseline_px,
            "results": results,
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nJSON written to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
