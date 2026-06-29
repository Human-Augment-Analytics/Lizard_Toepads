import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ultralytics import YOLO
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import logging
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from datetime import datetime

from common.obb_utils import order_box_points, crop_obb_from_corners, transform_keypoints, crop_toe_boxes_obb

MODEL_NAME = "hrnet_gcn"
SCRIPT_DIR = Path(__file__).parent.resolve()

def setup_logging():
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(str(log_dir / f"{MODEL_NAME}.log")),
            logging.StreamHandler(sys.stdout),
        ]
    )

def make_chain_edge_index(num_landmarks=9):
    edges = []
    for i in range(num_landmarks - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index

def landmark_loss(pred_coords, gt_coords):
    coord_loss = F.mse_loss(pred_coords, gt_coords)
    pred_dists = (pred_coords[:, 1:] - pred_coords[:, :-1]).norm(dim=-1)
    
    gt_dists = (gt_coords[:, 1:] - gt_coords[:, :-1]).norm(dim=-1)
    dist_loss = F.mse_loss(pred_dists, gt_dists)
    
    return coord_loss + 0.5 * dist_loss
    

def visualize_landmarks(img_tensor, pred_coords, gt_coords=None, save_path=None):
    img = img_tensor.permute(1,2,0).cpu().numpy()
    img = (img * 255).astype('uint8')
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    
    pred_xy = pred_coords.cpu().numpy() * img.shape[1]
    plt.scatter(pred_xy[:,0], pred_xy[:,1], c='r', label='pred', s=30)
    
    if gt_coords is not None:
        gt_xy = gt_coords.cpu().numpy() * img.shape[1]
        plt.scatter(gt_xy[:, 0], gt_xy[:, 1], c='g', label="gt", s=30, marker='x')
        
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
def compute_rescaled_pixel_error(pred_coords, coords, orig_size, device="cuda"):
    # coords are normalized [0,1], so convert to 512 space
    pred_px = pred_coords * 512
    gt_px = coords * 512
    orig_h = orig_size[:, 0]  # shape: (B,)
    orig_w = orig_size[:, 1]  # shape: (B,)
    # compute per-axis scale
    scale_x = (512 / orig_w).to(device)
    scale_y = (512 / orig_h).to(device)
    
    dx = (pred_px[:,:,0] - gt_px[:,:,0]) / scale_x.unsqueeze(1)
    dy = (pred_px[:,:,1] - gt_px[:,:,1]) / scale_y.unsqueeze(1)

    error = torch.sqrt(dx**2 + dy**2)  # (B, num_points)

    return error.mean(dim=1).sum().item()

def train(model, train_dataset, val_dataset=None, device='cuda', epochs=10, batch_size=4, lr=1e-4, vis_path="./"):
    setup_logging()
    model = model.to(device)
    model.train()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )
    edge_index = make_chain_edge_index(num_landmarks=train_dataset[0][1].shape[0]).to(device)
    best_val = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, coords, _ in train_loader:
            imgs = imgs.to(device)
            coords = coords.to(device)
            B = imgs.shape[0]
            #initial_coords = coords.clone() + ( torch.rand_like(coords)  * 0.03 )
            mean_shape = torch.tensor([
                [0.3, 0.9], [0.4, 0.8], [0.5, 0.7],
                [0.6, 0.6], [0.7, 0.5], [0.8, 0.4],
                [0.7, 0.3], [0.6, 0.2], [0.5, 0.1],
            ], dtype=torch.float).to(imgs.device)

            initial_coords = mean_shape.unsqueeze(0).repeat(B,1,1)
            
            pred_coords = model(imgs, initial_coords, edge_index)
            loss = landmark_loss(pred_coords, coords)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * imgs.size(0)
            
        epoch_loss /= len(train_dataset)        
        val_loss = None
        pix_err = None
        if val_dataset is not None:
            model.eval()
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            val_loss_total = 0.0
            pxerr_total = 0.0
            with torch.no_grad():
                for imgs, coords, orig_size in val_loader:
                    imgs = imgs.to(device)
                    coords = coords.to(device)
                    B = imgs.shape[0]
                    #initial_coords = coords.clone()
                    mean_shape = torch.tensor([
                        [0.3, 0.9], [0.4, 0.8], [0.5, 0.7],
                        [0.6, 0.6], [0.7, 0.5], [0.8, 0.4],
                        [0.7, 0.3], [0.6, 0.2], [0.5, 0.1],
                    ], dtype=torch.float).to(imgs.device)

                    initial_coords = mean_shape.unsqueeze(0).repeat(B,1,1)
                    pred_coords = model(imgs, initial_coords, edge_index)
                    val_loss_total += landmark_loss(pred_coords, coords).item() * imgs.size(0)
                    pxerr_total += compute_rescaled_pixel_error(pred_coords, coords, orig_size, device)
                    
            val_loss = val_loss_total / len(val_dataset)
            pix_err = pxerr_total / len(val_dataset)
            
            if val_loss < best_val:
                best_val = val_loss
                ckpt_dir = SCRIPT_DIR / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(ckpt_dir / f"{MODEL_NAME}_best.pth"))
            
            scheduler.step(val_loss)
            model.train()
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}, Avg Pixel Error: {pix_err}" +
             (f", Val Loss: {val_loss:.6f}" if val_loss is not None else ""))
        
        if epoch % 10 == 0:
            vp = f"{vis_path}/visualization_epoch{epoch}.jpg"
            visualize_landmarks(imgs[0], pred_coords[0], coords[0], save_path=vp)
        
    return model
    