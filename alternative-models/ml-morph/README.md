# ml-morph (dlib Shape Predictor)

Cascade shape regression model using dlib's ensemble of regression trees. This is the baseline model that the deep learning alternatives (Stacked Hourglass, ViT, HRNet, HRNet-GCN) are compared against.

## Architecture

- **Method**: Ensemble of regression trees (cascade shape regression)
- **Library**: dlib (`dlib.shape_predictor`)
- **Input**: 512×512 BGR crop images with a full-image bounding box
- **Output**: 9 landmark (x, y) coordinates in pixel space

## Dependencies

Requires `dlib` (which needs cmake/boost to build from source):

```bash
uv pip install dlib
```

On PACE, cmake and boost are typically available via module system.

## Training

```bash
cd alternative-models/ml-morph
python train.py --split ../benchmarking/splits/split.json
```

This will:
1. Convert shared `.pt` crop files to `.jpg` images + dlib XML format
2. Train the shape predictor using `dlib.train_shape_predictor`
3. Evaluate on the validation split and print mean pixel deviation
4. Save the predictor to `checkpoints/ml_morph_best.dat`

### Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--threads` | 8 | CPU threads for training |
| `--tree-depth` | 4 | Regression tree depth |
| `--cascade-depth` | 15 | Number of cascades |
| `--nu` | 0.1 | Regularization parameter |
| `--oversampling` | 20 | Oversampling amount |
| `--test-splits` | 20 | Number of test splits |
| `--feature-pool-size` | 500 | Feature pool size |
| `--num-trees` | 500 | Trees per cascade level |

## Evaluation

ml-morph is automatically included in `evaluate.py` when a trained checkpoint exists at `checkpoints/ml_morph_best.dat`. If dlib is not installed, it is skipped gracefully.

## Data Format

Training uses dlib's XML format. The `convert_to_xml.py` script bridges the shared `.pt` Crop_File format to dlib XML:

- `.pt` crops → `.jpg` images saved to `data/images/`
- Landmark coordinates → `<part>` elements in XML
- Bounding box → full 512×512 image (crops are already tightly framed)

## Source

Based on the ml-morph method: Porto & Voje (2020), "ML-morph: A fast, accurate and general approach for automated detection and landmarking of biological structures in images." Methods in Ecology and Evolution.
