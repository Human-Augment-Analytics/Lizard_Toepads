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

    dataset = LizardDataset(pt_paths, input_size=config.input_size)
    train_size = int(config.train_val_split * len(dataset))
    val_size = len(dataset) - train_size

    if args.split:
        if not Path(args.split).exists():
            print(f"ERROR: split file not found: {args.split}", file=sys.stderr)
            sys.exit(1)
        with open(args.split) as f:
            split_data = json.load(f)
        train_dataset = LizardDataset(split_data["train"], input_size=config.input_size)
        val_dataset = LizardDataset(split_data["val"], input_size=config.input_size)
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    model = HRNetGNN(hrnet_backbone="hrnet_w18", feat_dim=config.feat_dim, gnn_hidden=config.gnn_hidden,
                    num_layers=config.num_layers, num_landmarks=config.num_landmarks, num_iters=config.num_iters)

    trained_model = train(model, train_dataset, val_dataset=val_dataset,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        epochs=config.epochs, batch_size=config.batch_size, lr=config.lr)

if __name__ == "__main__":
    main()