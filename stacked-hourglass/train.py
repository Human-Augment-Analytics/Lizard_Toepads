from model.stackedhourglass import StackedHourGlass
from lizarddataset import LizardDataset
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
import json

def main(args):
    training_data_dir = args.data
    if training_data_dir is None:
        training_data_dir = "./data"

    configName = args.config
    config = loadConfig(configName)
    validateConfig(config) # fill any missing values
    print(config)
    initEnvironment()

    npz_dir = Path(f"{training_data_dir}/heatmaps")
    npz_paths = list(npz_dir.glob("*.npz"))
    print(f"Found {len(npz_paths)} training samples at location {npz_dir}")

    tsize = 1 - config["trainTestSplit"]
    train_paths, val_paths = train_test_split(npz_paths, test_size=tsize, random_state=config["randomState"])

    batch_size = config["batchSize"]

    dataset = LizardDataset(train_paths, aug_factor=config["augmentationFactor"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    valid_dataset = LizardDataset(val_paths, aug_factor=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shg = StackedHourGlass()
    shg.to(device)

    #optimizer = torch.optim.Adam(shg.parameters(), lr=config["initialLR"])
    optimizer = torch.optim.SGD(
        shg.parameters(),
        lr=1e-3,
        momentum=0.9,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config["scheduler"]["factor"], patience=config["scheduler"]["patience"]
    )
    num_epochs = config["epochs"]
    best_val_loss = 10
    for epoch in range(num_epochs):
        shg.train()
        running_loss = 0.0

        total = len(dataloader)

        batchct = 0
        for imgs, gt_heatmaps in dataloader:
            starttime = time.time()
            imgs, gt_heatmaps = imgs.to(device), gt_heatmaps.to(device)
            optimizer.zero_grad()
            combined_hm_preds = shg(imgs)
            pred_list = [combined_hm_preds[:, i, :, :, :] for i in range(combined_hm_preds.shape[1])] #hack
            loss = shg.calc_loss(pred_list, gt_heatmaps).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batchct += 1
            endtime = time.time()
            runtime = endtime-starttime
            print(f"Batch {batchct} / {total} | Process Time: {runtime} s | ETA: {(total-batchct)*runtime} | Loss: {loss.item()}", end="\r", flush=True)
        avg_train_loss = running_loss / len(dataloader)
        
        # -------------------------
        # Validation
        # -------------------------
        shg.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, gt_heatmaps in valid_dataloader:
                imgs, gt_heatmaps = imgs.to(device), gt_heatmaps.to(device)
                preds = shg(imgs)
                loss = shg.calc_loss(preds, gt_heatmaps).mean()
                val_loss += loss.item()
        avg_val_loss = val_loss / len(valid_dataloader)
        scheduler.step(avg_val_loss)
        checkpoint_path = f"checkpoints/shg_epoch_best.pth"
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(shg.state_dict(), checkpoint_path)
        print()
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

def initEnvironment():
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

def loadConfig(cname):
    if cname != None:
        p = Path(f"./configs/{cname}.json")
        if p.exists():
            try:
                with open(p, "r") as f:
                    config = json.load(f)
                    return config
            except Exception as e:
                print(f"Unable to load config {cname} at path {p}")
    else:
        return loadDefaultConfig()
    return None

def loadDefaultConfig():
    p = Path(f"./configs/default.json")
    with open(p, "r") as f:
        config = json.load(f)
        return config

def validateConfig(config):
    default = loadDefaultConfig()
    for key in default:
        if key not in config:
            config[key] = default[key]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stacked Hourglass")

    parser.add_argument("--config", type=str, required=False, help="Name of config file to use in config directory")
    parser.add_argument("--data", type=str, required=False, help="Path to training data directory")

    args = parser.parse_args()
    main(args)