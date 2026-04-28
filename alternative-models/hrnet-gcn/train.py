from utils import train
from lizard_dataset import LizardDataset
from hrnet_gcn import HRNetGNN
from config import HRNetGCNTrainingConfig

from torch.utils.data import DataLoader, Dataset, random_split
import torch
import os
import json
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Name of config file to use", default="default-config.json")
    parser.add_argument("--split", type=str, required=False, default=None)
    args = parser.parse_args()

    config = HRNetGCNTrainingConfig(args.config)

    DATA_PATH = config.training_data_path
    pt_paths = [os.path.join(DATA_PATH,f) for f in os.listdir(DATA_PATH) if f.endswith(".pt")]

    dataset = LizardDataset(pt_paths, input_size=512)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    if args.split:
        if not Path(args.split).exists():
            print(f"ERROR: split file not found: {args.split}", file=sys.stderr)
            sys.exit(1)
        with open(args.split) as f:
            split_data = json.load(f)
        train_dataset = LizardDataset(split_data["train"], input_size=512)
        val_dataset = LizardDataset(split_data["val"], input_size=512)
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    model = HRNetGNN(hrnet_backbone="hrnet_w18", feat_dim=1024, gnn_hidden=128,
                    num_layers=3, num_landmarks=9, num_iters=6)

    trained_model = train(model, train_dataset, val_dataset=val_dataset,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        epochs=80, batch_size=32, lr=1e-4)

if __name__ == "__main__":
    main()