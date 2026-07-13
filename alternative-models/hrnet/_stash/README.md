# Stash — Superseded WFLW Heatmap Baseline (v1)

## What is here

These files are the first WFLW heatmap training implementation, stashed on 2026-07-13.
They are preserved for reference and must not be imported or modified.

- `train_heatmap_wflw.py` — original WFLW training script
- `wflw_heatmap_dataset.py` — original WFLW dataset class

## Why stashed

Root cause of underperformance: augmentation was applied *after* loading
pre-cropped 512x512 .pt files. Scale jitter applied at 256px immediately
hit black padding borders, providing little useful geometric variety.
The reference pipeline applies scale jitter and rotation to the 512px crop
before resizing to 256px, giving the model context-rich augmentation.

## Baseline result

- Training: 60 epochs, full WFLW training split (~6000 samples after val split)
- Best val NME: 0.0530
- Test NME: 0.0605 on official WFLW 2500-sample test set
- Paper target: 0.046

## Replacement

The active pipeline is:
- `alternative-models/hrnet/wflw_pt_dataset.py` — new dataset using reference crop()
- `alternative-models/hrnet/train_heatmap_wflw_ref.py` — reference training loop
- Run via: `python alternative-datasets/WFLW/run_wflw_heatmap.py --split ...`

The existing checkpoint `checkpoints/hrnet_heatmap_wflw_best.pth` documents
the baseline result and is not deleted.
