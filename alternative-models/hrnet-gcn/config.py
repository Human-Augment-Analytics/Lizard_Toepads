import os
import json

class HRNetGCNTrainingConfig:
    def __init__(self, config_path):
        with open(config_path) as f:
            data = json.load(f)
            self.training_data_path = data["training_data_path"]