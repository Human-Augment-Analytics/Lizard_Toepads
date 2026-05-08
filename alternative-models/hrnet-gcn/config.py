import os
import json

class HRNetGCNTrainingConfig:
    def __init__(self, config_path):
        with open(config_path) as f:
            data = json.load(f)
            self.training_data_path = data["training_data_path"]
            self.num_landmarks = data.get("num_landmarks", 9)
            self.feat_dim = data.get("feat_dim", 64)
            self.gnn_hidden = data.get("gnn_hidden", 128)
            self.num_layers = data.get("num_layers", 2)
            self.num_iters = data.get("num_iters", 3)
            self.input_size = data.get("input_size", 512)
            self.epochs = data.get("epochs", 80)
            self.batch_size = data.get("batch_size", 32)
            self.lr = data.get("lr", 1e-4)
            self.train_val_split = data.get("train_val_split", 0.8)
