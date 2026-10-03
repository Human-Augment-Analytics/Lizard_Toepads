# ID Card Parsing Task

## Overview
We have been tasked with setting up an automatic parsing pipeline for new lizard ID cards that contain additional fields.

## OBB Annotator

A lightweight desktop tool for drawing a single YOLO oriented bounding box (OBB)
on each ID card image and exporting a reconstruction-ready annotation record.

Launch from this directory:

```bash
python -m annotator
```

- Source images live in `data/` as `{id}.jpg` (git-ignored, never committed).
- Annotations are written to `annotations/{id}.json` (committable, no pixels).
- Each record stores the image id, target resolution, the forward/inverse
  affine transform (original <-> preview space), and the OBB in target-space
  coordinates, so the box can be reconstructed in original pixels later.

Dependencies: Pillow, numpy, and the standard-library Tkinter GUI. No GPU or
deep-learning framework required.

### Tests

Headless backend tests (no GUI needed):

```bash
python -m pytest tests -q
```
