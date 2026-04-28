import numpy as np
import torch
from torch import nn
import torch
import torch.nn.functional as F
import timm
from datetime import datetime
from torch_geometric.nn import GCNConv


class HRNetGNN(nn.Module):
    def __init__(self, hrnet_backbone, feat_dim=64, gnn_hidden=128, num_layers=2, num_landmarks=9, num_iters=3):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_iters = num_iters
        
        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=True,
            features_only=True
        )
        
        self.feat_dim = feat_dim
        self.backbone_out_idx = -1
        
        self.node_feat_proj = nn.Linear(self.backbone.feature_info[self.backbone_out_idx]['num_chs'], gnn_hidden)
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(GCNConv(gnn_hidden, gnn_hidden))
            
        self.delta_head = nn.Linear(gnn_hidden, 2)
        
    def sample_features(self, feat_map, coords):
        B, C, H, W = feat_map.shape
        N = coords.shape[1]
        grid = coords.clone()
        
        grid = (grid * 2) - 1
        grid = grid.unsqueeze(2)
        
        sampled = F.grid_sample(feat_map, grid, align_corners=True)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        return sampled
    
    def forward(self, x, initial_coords, edge_index):
        coords = initial_coords.clone()
        feat_maps = self.backbone(x)
        feat_map = feat_maps[self.backbone_out_idx]
        
        for _ in range(self.num_iters):
            node_feats = self.sample_features(feat_map, coords)
            node_feats = self.node_feat_proj(node_feats)
            node_feats = F.relu(node_feats)
            B, N, F_dim = node_feats.shape
            node_feats_flat = node_feats.view(B*N, F_dim)
            batch_edge_index = []
            for b in range(B):
                batch_edge_index.append(edge_index + b*N)
            batch_edge_index = torch.cat(batch_edge_index, dim=1)
            
            h = node_feats_flat
            for layer in self.gnn_layers:
                h = layer(h, batch_edge_index)
                h = F.relu(h)
                
            delta = self.delta_head(h)
            delta = delta.view(B, N, 2)
            
            coords = coords + delta
            
        return coords