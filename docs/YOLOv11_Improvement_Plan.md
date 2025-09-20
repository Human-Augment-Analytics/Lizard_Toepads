## YOLOv11 Improvement Plan — Lizard Toepads (Small-Object Focus)

### Objectives
- Improve small-object detection (toepads/toes) accuracy without sacrificing reproducibility.
- Prioritize changes that are simple to operationalize on PACE; stage advanced architecture edits later.

### Small-Object Challenges (Context)
- Extreme downsampling: small objects become 1–4 px at low input sizes; features get lost in early strides.
- Heavy augmentations (Mosaic/MixUp) can erase tiny instances.
- Class imbalance: many images may contain few or no toes; sampler bias hurts recall.

### Prioritized Roadmap (Phased)
1) Baseline and High-Resolution Inputs (P0–P1)
   - Establish a clean 640 baseline with current pipeline and frozen seeds.
   - Train at higher resolution: imgsz=1280 (or 1024 if GPU-limited) to preserve small details.

2) Data Augmentation and Sampling (P1)
   - Reduce Mosaic/MixUp strength, enable Copy-Paste with curated crops.
   - Add gentle geometric and photometric jitters; avoid over-rotation/shear.
   - Oversample images containing toes to balance minibatches.

3) Tiling for Training and Inference (P2)
   - Slice large images into overlapping tiles to magnify local content; stitch predictions.
   - Train either on tiles or mixed (tiles + full frames).

4) Model/Architecture Tweaks (P3)
   - Use larger YOLOv11 variants (s/m) or custom YAML with an extra high-res detection head.
   - Explore attention blocks in neck (e.g., CBAM/SE) and BiFPN-style aggregation via custom model YAML.

5) Alternative Baseline for Cross-Check (P4)
   - Train Detectron2 Faster R-CNN + FPN baseline for a stability reference on small objects.

### Actionable Settings (Ultralytics CLI Examples)
Baselines (frozen recipe):
```bash
# P0: 640 baseline
yolo task=detect mode=train model=yolov11n.pt data=custom_data.yaml epochs=100 imgsz=640 \
  batch=16 optimizer=sgd cos_lr=True seed=42 workers=8 \
  mosaic=0.2 mixup=0.05 copy_paste=0.3 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=3 scale=0.15 translate=0.05 shear=0.0 perspective=0.0 \
  fliplr=0.0 flipud=0.0 \
  patience=50 \
  project=runs/detect name=baseline_640

# P1: High-res 1280
yolo task=detect mode=train model=yolov11s.pt data=custom_data.yaml epochs=100 imgsz=1280 \
  batch=8 optimizer=sgd cos_lr=True seed=42 workers=8 \
  mosaic=0.15 mixup=0.03 copy_paste=0.3 \
  degrees=3 scale=0.15 translate=0.05 \
  project=runs/detect name=hires_1280
```

Notes:
- Start with `yolov11s.pt` for 1280 if GPU memory is tight on `yolov11m/l/x`.
- Use `close_mosaic=15` (if available in your version) to disable mosaic late in training for stability.

### Tiling Strategy
When toes are extremely small, tiling can provide 2–8× local magnification without increasing model stride.

Training tiling (data prep):
- Add `scripts/preprocessing/tile_dataset.py` to produce overlapping tiles and remapped labels.
- Recommended defaults:
  - Tile size: 768 or 1024
  - Overlap: 15–25% (to avoid cutting objects)
  - Drop tiles without labels if class imbalance is severe, or keep with lower weight

Inference tiling:
- Add `scripts/inference/predict_tiled.py` to tile, predict per tile, de-duplicate with NMS across overlaps, and stitch to full-image coords.
- Use soft-NMS or DIoU-NMS with a slightly higher IoU threshold (e.g., 0.6–0.7) for cross-tile merging.

Potential pitfalls:
- Label duplication at tile borders; ensure proper clipping and mapping.
- Double counting: apply cross-tile NMS.

### Data Augmentation Guidelines (Small-Object Safe)
- Mosaic: 0.1–0.3 (avoid 0.5–1.0; can erase tiny objects)
- MixUp: 0–0.1 (light)
- Copy-Paste: 0.2–0.5 using curated toe/foot crops to increase diversity
- Geometric: degrees ≤5, scale ±0.2, translate ≤0.1, shear 0
- Color: moderate HSV jitter; avoid extreme brightness/contrast shifts
- Close to convergence: turn off mosaic/mixup in the last ~10–20 epochs

### Resampling Strategy (Class/Instance Imbalance)
Options (pick one to start):
1) Data duplication: duplicate lines in custom train list to increase sampling of toe-rich images.
2) Weighted sampler: implement a small wrapper around Ultralytics dataloader to pass a `WeightedRandomSampler` based on per-image toe count.
3) Two-stage: prefilter random minibatches to require ≥1 labeled object.

Minimal approach to start: curate a `train.txt` file listing images and duplicate paths 1–3× for toe-rich samples.

### Architecture Tweaks (Advanced)
- Larger models first: `yolov11m/l` with imgsz 1280.
- Extra small-object head: edit model YAML to add detection on a higher-resolution pyramid level (e.g., P2/P3) to reduce effective stride.
- Attention in neck: inject SE/CBAM blocks in the PAN/FPN layers (custom YAML + custom layers file).
- BiFPN: craft a custom model YAML replacing PAN with BiFPN-style top-down/bottom-up merges.

These require custom YAMLs and possibly custom modules; keep them in `models/` and document thoroughly.

### Alternative Baseline: Detectron2 (Reference)
Pros: FPN is very stable for small objects; rich configs and augmentations. Cons: heavier and slower; new framework.
- Train Faster R-CNN + FPN on the same splits; compare mAP and small-object recall.
- If masks are available, try Mask R-CNN for better localization around toes.

### Resource Planning on PACE
- Start with `H200:1` GPU and batch sizes tuned to memory limits (monitor OOM).
- For imgsz=1280 on `yolov11s`, expect batch ≈ 8–16 depending on GPU.
- Prefer local scratch for datasets and `runs/` to avoid home quota pressure.

### Milestones & Exit Criteria
- P1 complete when hires run (1280) exceeds 640 baseline mAP by ≥3 points and improves small-object recall.
- P2 complete when tiling adds ≥1–2 mAP and improves precision at small sizes without major FP rate increase.
- P3 complete when custom head/neck beats P2 by ≥1 mAP with acceptable latency.

### Next Steps (Immediate)
1) Run hires_1280 training with reduced mosaic/mixup.
2) Prepare tiler for training and inference; validate label mapping visually.
3) Add simple oversampling via duplicated train entries.


