"""Overlay GT landmarks (numbered) + cephalometric graph edges on one sample.

Loads a single preprocessed cephalometric .pt file and renders the 19
ground-truth landmarks with their indices and the default anatomical graph
edges, so you can eyeball whether the topology and landmark ordering look right.

Usage:
    python -m landmarking.scripts.visualize_cephalometric_graph \
        --pt /home/.../Cephalometric_data/train/001.pt \
        --out ceph_overlay.png

    # or just point at a directory and it grabs the first .pt
    python -m landmarking.scripts.visualize_cephalometric_graph \
        --dir /home/.../Cephalometric_data/train
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt

from ..common.graph_topologies import make_cephalometric_edge_index

LANDMARK_NAMES = [
    "Sella", "Nasion", "Orbitale", "Porion", "A-point", "B-point", "Pogonion",
    "Menton", "Gnathion", "Gonion", "L1", "U1", "UpperLip", "LowerLip",
    "Subnasale", "SoftPog", "PNS", "ANS", "Articulare",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Visualize cephalometric GT + graph.")
    ap.add_argument("--pt", type=str, default=None, help="Path to a single .pt file.")
    ap.add_argument("--dir", type=str, default=None, help="Dir; uses first *.pt.")
    ap.add_argument("--out", type=str, default="ceph_overlay.png", help="Output PNG.")
    args = ap.parse_args(argv)

    if args.pt:
        pt_path = Path(args.pt)
    elif args.dir:
        pts = sorted(Path(args.dir).glob("*.pt"))
        if not pts:
            print(f"No .pt files in {args.dir}", file=sys.stderr)
            sys.exit(1)
        pt_path = pts[0]
    else:
        print("Provide --pt or --dir", file=sys.stderr)
        sys.exit(1)

    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    img = data["image"]  # (3, H, W) uint8
    if hasattr(img, "numpy"):
        img = img.numpy()
    img_hw = np.transpose(img, (1, 2, 0))  # HWC for display
    H, W = img_hw.shape[:2]

    coords = data["tps"].numpy()  # (19, 2) normalized [0,1]
    pts_px = coords.copy()
    pts_px[:, 0] *= W  # x
    pts_px[:, 1] *= H  # y

    # Edges (2, E) directed; dedupe to undirected for drawing.
    ei = make_cephalometric_edge_index().numpy()
    undirected = {tuple(sorted((int(ei[0, k]), int(ei[1, k])))) for k in range(ei.shape[1])}

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img_hw)

    # Draw edges
    for u, v in sorted(undirected):
        ax.plot(
            [pts_px[u, 0], pts_px[v, 0]],
            [pts_px[u, 1], pts_px[v, 1]],
            "-", color="cyan", linewidth=1.0, alpha=0.7,
        )

    # Draw landmarks + numbers
    ax.scatter(pts_px[:, 0], pts_px[:, 1], s=30, c="red", zorder=3)
    for i, (x, y) in enumerate(pts_px):
        ax.text(
            x + 6, y - 6, str(i), color="yellow", fontsize=11, weight="bold", zorder=4,
        )

    ax.set_title(f"{pt_path.name}  ({len(undirected)} edges, {len(pts_px)} landmarks)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"Saved overlay to {args.out}")
    print("Landmark legend (index: name):")
    for i, name in enumerate(LANDMARK_NAMES):
        print(f"  {i:2d}: {name}")


if __name__ == "__main__":
    main()
