"""Report lizard crop-size / resolution stats to decide input_size.

For each preprocessed lizard .pt, reconstructs the pre-letterbox crop size
from the stored `scale` (letterbox resize factor) and reports:
  - crop long-side pixel distribution (before the 512 downsample)
  - effective downsample factor at 512 / 768 / 1024
  - mm-per-pixel at each input_size (using ruler_px + ruler_mm=10)

This tells you whether 512 is discarding annotation-relevant detail.

Usage:
    python -m landmarking.scripts.lizard_resolution_report \
        --dir /home/.../Lizard_data/lizard/train
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def _to_float(v, default=None):
    if v is None:
        return default
    if hasattr(v, "item"):
        try:
            return float(v.item())
        except Exception:
            pass
    try:
        return float(v)
    except Exception:
        return default


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lizard resolution diagnostic.")
    ap.add_argument("--dir", type=str, required=True, help="Dir of lizard .pt files.")
    ap.add_argument("--canvas", type=int, default=512, help="Current stored canvas size.")
    ap.add_argument("--ruler-mm", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0, help="Max files (0 = all).")
    args = ap.parse_args(argv)

    paths = sorted(Path(args.dir).glob("*.pt"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"No .pt files in {args.dir}", file=sys.stderr)
        sys.exit(1)

    crop_long = []          # reconstructed pre-letterbox crop long side (px)
    mmpp_512, mmpp_768, mmpp_1024 = [], [], []
    n_ruler = 0

    for p in paths:
        d = torch.load(str(p), map_location="cpu", weights_only=False)
        scale = _to_float(d.get("scale"))
        # Letterbox: crop was resized by `scale` to fit `canvas`. So the crop
        # long side ~= canvas / scale (scale <= 1 for downsizing large crops).
        if scale and scale > 0:
            crop_long.append(args.canvas / scale)

        ruler_px = _to_float(d.get("ruler_px"))
        if ruler_px and ruler_px > 0:
            # ruler_px is measured in the 512 canvas space. mm-per-canvas-pixel:
            mm_per_canvas_px = args.ruler_mm / ruler_px
            # Physical mm per model pixel scales inversely with input_size.
            mmpp_512.append(mm_per_canvas_px * (args.canvas / 512.0))
            mmpp_768.append(mm_per_canvas_px * (args.canvas / 768.0))
            mmpp_1024.append(mm_per_canvas_px * (args.canvas / 1024.0))
            n_ruler += 1

    def stats(a):
        if not a:
            return "n/a"
        a = np.array(a)
        return (f"min={a.min():.3g} p25={np.percentile(a,25):.3g} "
                f"med={np.median(a):.3g} p75={np.percentile(a,75):.3g} "
                f"max={a.max():.3g}")

    print(f"Files analyzed: {len(paths)}  (with ruler: {n_ruler})")
    print(f"Stored canvas size: {args.canvas}")
    print()
    print("Pre-letterbox crop LONG SIDE (px), reconstructed from `scale`:")
    print("  " + stats(crop_long))
    if crop_long:
        cl = np.array(crop_long)
        print(f"  -> downsample factor to 512:  med={np.median(cl)/512.0:.2f}x")
        print(f"  -> downsample factor to 768:  med={np.median(cl)/768.0:.2f}x")
        print(f"  -> downsample factor to 1024: med={np.median(cl)/1024.0:.2f}x")
    print()
    print("mm per MODEL pixel (lower = finer resolvable detail):")
    print(f"  @512:  {stats(mmpp_512)}")
    print(f"  @768:  {stats(mmpp_768)}")
    print(f"  @1024: {stats(mmpp_1024)}")
    print()
    print("Interpretation:")
    print("  - If crop long side >> input_size, you are downsampling real detail;")
    print("    a larger input_size lowers mm/pixel and the error floor.")
    print("  - If crop long side <= input_size, 512 is already upsampling and a")
    print("    bump only adds compute, not precision.")


if __name__ == "__main__":
    main()
