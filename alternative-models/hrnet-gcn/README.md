# HRNet-GCN

## Architecture

HRNet backbone combined with a Graph Convolutional Network (GCN) head for landmark detection. This is the winning architecture that outperformed the ml-morph baseline (cascade shape regression with decision trees).

## Sources

- Formalized from: `stacked-hourglass/hrnet/hrnet_gnn.ipynb`
- Original formalization: `HRNet-GCN/` (moved here)

## Files

- `hrnet_gcn.py` — model definition
- `lizard_dataset.py` — dataset loader
- `train.py` — training script
- `config.py` — configuration handling
- `default-config.json` — default hyperparameters
- `utils.py` — utility functions

## How to Run

```bash
cd alternative-models/hrnet-gcn

python train.py
```

See `default-config.json` for configurable paths and hyperparameters.
