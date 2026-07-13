"""
WFLWPtDataset — reference pipeline dataset for HRNet heatmap regression.

Reads our existing pre-cropped 512x512 .pt files and presents the reference
(img_tensor, target_hm, meta) interface expected by train_heatmap_wflw_ref.py.

Key design:
- Uses crop_compat() — a NumPy-2.0-safe reimplementation of the reference
  crop() that uses cv2 throughout instead of scipy.misc / np.math.
  The reference crop() has NumPy 2.0 incompatibilities (np.math removed)
  and scipy.misc incompatibilities (imresize/imrotate removed in SciPy 1.3).
  crop_compat() produces geometrically identical results.
- generate_target() and transform_pixel() from the reference repo are imported
  directly (they have no compatibility issues).
- Fixed center=(256, 256) and scale=512/200=2.56 — valid because all .pt files
  are pre-cropped 512x512 affine squares.
- Augmentation: flip (p=0.5), scale jitter (+-25%), rotation (+-30 deg, p=0.6)
  applied to the 512px image before crop resizes to 256x256, recovering
  the context-rich augmentation behaviour of the reference pipeline.
- meta dict contains center, scale, pts (512px) for decode_preds + compute_nme.

Returns: (img_tensor, target_hm, meta)
  img_tensor: (3, 256, 256) float32, ImageNet-normalised
  target_hm:  (98, 64, 64) float32, Gaussian heatmap targets (sigma=1.5)
  meta:       dict with index, center (Tensor[2]), scale (float),
              pts (Tensor[98,2] in 512px space), tpts (Tensor[98,2] in 64px space)
"""
import sys
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data

# ── Import reference functions from the patched HRNet repo ───────────────────
# Resolve relative to this file's location so the path works regardless of cwd.
_HERE = Path(__file__).resolve().parent
_REF_REPO = _HERE.parent.parent.parent / "HRNet-Facial-Landmark-Detection"

# Fallback: walk up from cwd if __file__ resolution doesn't reach the workspace root
if not _REF_REPO.exists():
    _cwd = Path.cwd()
    for _candidate in [_cwd, _cwd.parent, _cwd.parent.parent]:
        _try = _candidate / "HRNet-Facial-Landmark-Detection"
        if _try.exists():
            _REF_REPO = _try
            break

# Add repo root (not lib/) so that relative imports inside the package resolve.
_REF_ROOT = str(_REF_REPO)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

# Import only the functions that are NumPy 2.0 safe.
# We do NOT import crop() — it uses np.math (removed in NumPy 2.0) and
# scipy.misc (removed in SciPy 1.3). We use crop_compat() below instead.
from lib.utils.transforms import (
    generate_target,
    fliplr_joints,
    transform_pixel,
    get_affine_transform,
)

# ── Constants ─────────────────────────────────────────────────────────────────
_FIXED_CENTER = torch.Tensor([256.0, 256.0])
_FIXED_SCALE  = 512.0 / 200.0   # = 2.56

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_INPUT_SIZE    = [256, 256]
_HEATMAP_SIZE  = [64, 64]
_SIGMA         = 1.5
_NUM_LANDMARKS = 98


def crop_compat(img, center, scale, output_size, rot=0):
    """NumPy-2.0-safe replacement for the reference crop() function.

    The reference crop() uses np.math.floor (removed in NumPy 2.0) and
    scipy.misc.imresize/imrotate (removed in SciPy 1.3). This implementation
    is geometrically identical but uses only cv2 and standard math.

    For our use case (512px input, scale=2.56, output_size=[256,256]):
      sf = scale * 200 / output_size[0] = 2.56 * 200 / 256 = 2.0
    This hits the sf >= 2 branch in the original, which downsizes the source
    image before extracting the crop. We replicate that exactly.

    Args:
        img:         (H, W, 3) float32 HWC array
        center:      torch.Tensor([cx, cy]) or numpy array in pixel space
        scale:       float — face size = scale * 200 pixels
        output_size: [W, H] of the output crop (both 256 for us)
        rot:         rotation angle in degrees (0 for val/test)

    Returns:
        (output_size[1], output_size[0], 3) uint8 HWC array
    """
    # get_affine_transform expects a numpy array, not a torch.Tensor
    if hasattr(center, 'numpy'):
        center_np = center.numpy().astype(np.float32)
    else:
        center_np = np.array(center, dtype=np.float32)

    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / output_size[0]

    if sf >= 2:
        # Downsample source image for efficiency (matches reference behaviour)
        new_ht = int(math.floor(ht / sf))
        new_wd = int(math.floor(wd / sf))
        if max(new_ht, new_wd) < 2:
            return np.zeros((output_size[1], output_size[0], img.shape[2]),
                            dtype=np.uint8)
        img = cv2.resize(img.astype(np.uint8), (new_wd, new_ht),
                         interpolation=cv2.INTER_LINEAR)
        center_np = center_np / sf
        scale = scale / sf

    trans = get_affine_transform(center_np, np.array([scale, scale], dtype=np.float32), rot, output_size)
    dst = cv2.warpAffine(
        img.astype(np.uint8), trans,
        (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return dst


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

        # ── Step 4: Crop to 256x256 using crop_compat() ─────────────────
        # crop_compat() is our cv2-based replacement for the reference crop()
        # which has NumPy 2.0 (np.math) and SciPy (scipy.misc) incompatibilities.
        img_cropped = crop_compat(img, center, scale, _INPUT_SIZE, rot=r)
        # img_cropped: (256, 256, 3) uint8

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
