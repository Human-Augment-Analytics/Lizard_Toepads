## Benchmarking Strategy — Lizard Toepads Detection

### Goals
- Provide a fair, reproducible yardstick for small-object detection performance.
- Compare within-YOLO changes (imgsz, tiling, aug) and against a Detectron2 baseline.

### Datasets & Splits
- Source: `dataset/` with `images/{train,val}` and `labels/{train,val}`.
- Fixed seed split (e.g., 80/20) and frozen file lists committed under `docs/splits/` if possible.
- Optional test split: if long-run comparison needed.

### Metrics
- Primary: mAP@[0.50:0.95] (COCO-style), mAP@0.5.
- Small-object sensitivity: report mAP_small or approximate by filtering objects with area below a threshold; also report Recall@0.5 for small objects.
- Per-class AP: toe vs toepad vs ruler (if applicable).
- Error analysis: FP types (ruler confusion, background) and FN rate on toes.

### Evaluation Protocol
1) Standard validation on the held-out val set using Ultralytics `val`.
2) Tiled-inference evaluation: if training uses tiling, run evaluation with the same tiling inference wrapper to avoid scale mismatch.
3) Confidence thresholds: sweep or fix at 0.25; report PR curves and calibrate a working threshold.
4) NMS: record IoU threshold; for tiled inference consider higher IoU (0.6–0.7) to merge cross-tile duplicates.

Ultralytics examples:
```bash
# Standard eval
yolo task=detect mode=val model=runs/detect/hires_1280/weights/best.pt data=custom_data.yaml imgsz=1280 \
  conf=0.25 iou=0.6 save_json=True plots=True

# Predict (qualitative)
yolo task=detect mode=predict model=runs/detect/hires_1280/weights/best.pt source=dataset/images/val \
  imgsz=1280 conf=0.25 save=True
```

### Experiment Matrix (Ablations)
- Input size: 640 vs 1024 vs 1280
- Augmentation: baseline vs reduced mosaic/mixup vs +copy-paste
- Tiling: off vs train-tiling vs infer-tiling
- Model size: yolov11n/s/m
- Head tweak: default vs extra high-res head (if implemented)

Run one change at a time; keep seeds fixed.

### Logging & Reproducibility
- Record `ultralytics.yaml` effective cfg, seeds, GPU type, CUDA/driver versions.
- Save `runs/detect/<exp>/results.csv`, PR curves, and confusion matrices.
- Maintain a `docs/benchmarks.md` table with key results (see template below).

### Result Table Template
| ID | Change | imgsz | Tile | Aug | Model | mAP@[.5:.95] | mAP@.5 | Recall | Notes |
|----|--------|-------|------|-----|-------|--------------|--------|--------|-------|
| B1 | Baseline | 640 | No | base | v11n |  |  |  |  |
| H1 | High-res | 1280 | No | mild | v11s |  |  |  |  |
| T1 | +Tiling (infer) | 1280 | Yes | mild | v11s |  |  |  |  |
| A1 | +Copy-Paste | 1280 | Yes | cp | v11s |  |  |  |  |
| D2 | Detectron2 FRCNN-FPN | 1024 | N/A | mild | R50-FPN |  |  |  |  |

### Qualitative Review
- Save `top_k` FP and FN examples for weekly review.
- Visualize cross-tile merges to ensure no duplicate boxes.

### Detectron2 Baseline (Optional)
- Train Faster R-CNN + FPN with small-object-friendly anchors or higher-res inputs.
- Evaluate with the same splits and report mAP.

### Decision Criteria
- Adopt a change when it improves mAP@[.5:.95] by ≥1 point and small-object recall by ≥2 points without >20% latency penalty at inference resolution.


