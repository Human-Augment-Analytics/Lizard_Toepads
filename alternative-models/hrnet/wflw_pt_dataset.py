"""
WFLWPtDataset — reference pipeline dataset for HRNet heatmap regression.

Reads our existing pre-cropped 512x512 .pt files and presents the reference
(img_tensor, target_hm, meta) interface expected by train_heatmap_wflw_ref.py.

Key design:
- Uses the patched reference crop(), generate_target(), fliplr_joints(),
  transform_pixel() from HRNet-Facial-Landmark-Detection/lib/utils/transforms.py
- Fixed center=(256, 256) and scale=512/200=2.56 — valid because all .pt files
  are pre-cropped 512x512 affine squares
- Augmentation: flip (p=0.5), scale jitter (+-25%), rotation (+-30 deg, p=0.6)
  applied to the 512px image before crop() resizes to 256x256, recovering
  the context-rich augmentation behaviour of the reference pipeline
- meta dict contains center, scale, pts (512px) for decode_preds + compute_nme

Returns: (img_tensor, target_hm, meta)
  img_tensor: (3, 256, 256) float32, ImageNet-normalised
  target_hm:  (98, 64, 64) float32, Gaussian heatmap targets (sigma=1.5)
  meta:       dict with index, center (Tensor[2]), scale (float),
              pts (Tensor[98,2] in 512px space), tpts (Tensor[98,2] in 64px space)
"""
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

# ── Import reference functions from the patched HRNet repo ───────────────────
# Resolve relative to this file's location so the path works regardless of cwd.
_HERE = Path(__file__).resolve().parent
_REF_LIB = _HERE.parent.parent.parent / "HRNet-Facial-Landmark-Detection" / "lib"

# Fallback: walk up from cwd if __file__ resolution doesn't reach the workspace root
if not _REF_LIB.exists():
    _cwd = Path.cwd()
    for _candidate in [_cwd, _cwd.parent, _cwd.parent.parent]:
        _try = _candidate / "HRNet-Facial-Landmark-Detection" / "lib"
        if _try.exists():
            _REF_LIB = _try
            break

if str(_REF_LIB) not in sys.path:
    sys.path.insert(0, str(_REF_LIB))

from utils.transforms import crop, generate_target, fliplr_joints, transform_pixel

# ── Constants ─────────────────────────────────────────────────────────────────
# All .pt crops are 512x512 affine-cropped squares.
# center = geometric image centre; scale so that scale*200 = 512 px.
_FIXED_CENTER = torch.Tensor([256.0, 256.0])
_FIXED_SCALE  = 512.0 / 200.0   # = 2.56

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_INPUT_SIZE   = [256, 256]   # crop() output size
_HEATMAP_SIZE = [64, 64]     # generate_target() output size
_SIGMA        = 1.5          # Gaussian sigma in heatmap pixels (paper: 1.5)
_NUM_LANDMARKS = 98


class WFLWPtDataset(data.Dataset):
    """WFLW dataset backed by pre-cropped .pt files.

    Uses the reference HRNet augmentation pipeline (crop, fliplr_joints,
    generate_target, transform_pixel) to match paper training conditions.

    Args:
        pt_paths:     List of .pt file path strings.
        augment:      Whether to apply augmentation (True for train, False for val/test).
        flip_prob:    Probability of horizontal flip (default 0.5, matches paper).
        scale_factor: Scale jitter range; scale drawn from
                      Uniform(1-scale_factor, 1+scale_factor) (default 0.25).
        rot_factor:   Max rotation angle in degrees; applied with prob 0.6
                      (default 30, matches paper ROT_FACTOR: 30).
    """

    def __init__(self, pt_paths, augment=True, flip_prob=0.5,
                 scale_factor=0.25, rot_factor=30):
        self.paths        = list(pt_paths)
        self.augment      = augment
        self.flip_prob    = flip_prob
        self.scale_factor = scale_factor
        self.rot_factor   = rot_factor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pt_path = self.paths[idx]

        # ── Step 1: Load .pt file ────────────────────────────────────────
        try:
            data_dict = torch.load(pt_path, map_location="cpu")
        except Exception as e:
            raise IOError(f"Failed to load .pt file: {pt_path}") from e

        # img: (3, 512, 512) uint8 tensor → (512, 512, 3) uint8 HWC numpy
        img = data_dict["image"].permute(1, 2, 0).numpy().astype(np.float32)

        # tps: (98, 2) float32 [0,1] → 512px pixel space
        tps = data_dict["tps"].numpy().astype(np.float64)
        pts = tps * 512.0   # (98, 2) in pixel space

        # ── Step 2: Set fixed center and scale ───────────────────────────
        center = _FIXED_CENTER.clone()   # mutable copy — flip may update it
        scale  = _FIXED_SCALE            # may be jittered below

        # ── Step 3: Augmentation ─────────────────────────────────────────
        r = 0  # rotation angle (degrees)

        if self.augment:
            # Horizontal flip (p=flip_prob)
            if random.random() <= self.flip_prob:
                img    = np.fliplr(img)
                pts    = fliplr_joints(pts, width=512, dataset='WFLW')
                center[0] = 512 - center[0]

            # Scale jitter
            scale = scale * random.uniform(1 - self.scale_factor,
                                           1 + self.scale_factor)

            # Rotation (p=0.6, matching paper)
            if random.random() <= 0.6:
                r = random.uniform(-self.rot_factor, self.rot_factor)

        # ── Step 4: Crop to 256x256 using reference crop() ───────────────
        # crop() expects a float32 HWC array and a torch.Tensor center.
        img_cropped = crop(img, center, scale, _INPUT_SIZE, rot=r)
        # img_cropped: (256, 256, 3) float32

        # ── Step 5: Transform landmarks to 64px heatmap space ─────────────
        # transform_pixel maps a point from original image space to the
        # heatmap output space (64x64) using the same center/scale/rot as crop().
        tpts = pts.copy()
        for i in range(_NUM_LANDMARKS):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(
                    tpts[i, 0:2] + 1,   # +1: convert to 1-indexed (reference convention)
                    center, scale,
                    _HEATMAP_SIZE,
                    rot=r
                )

        # ── Step 6: Generate Gaussian heatmap targets ────────────────────
        target = np.zeros((_NUM_LANDMARKS,
                           _HEATMAP_SIZE[0],
                           _HEATMAP_SIZE[1]), dtype=np.float32)
        for i in range(_NUM_LANDMARKS):
            if tpts[i, 1] > 0:
                target[i] = generate_target(
                    target[i],
                    tpts[i] - 1,   # -1: back to 0-indexed for generate_target
                    _SIGMA
                )

        # ── Step 7: Normalise image ──────────────────────────────────────
        img_norm = img_cropped.astype(np.float32) / 255.0
        img_norm = (img_norm - _IMAGENET_MEAN) / _IMAGENET_STD
        img_tensor = torch.from_numpy(
            img_norm.transpose(2, 0, 1)    # HWC → CHW
        ).float()

        # ── Step 8: Build meta dict ──────────────────────────────────────
        # meta['pts'] must be in 512px space for decode_preds + compute_nme.
        meta = {
            "index": idx,
            "center": center,                          # Tensor[2], post-flip
            "scale": scale,                            # float, post-jitter
            "pts":   torch.Tensor(pts),                # (98,2) 512px, post-flip
            "tpts":  torch.Tensor(tpts),               # (98,2) 64px heatmap space
        }

        return img_tensor, torch.Tensor(target), meta
