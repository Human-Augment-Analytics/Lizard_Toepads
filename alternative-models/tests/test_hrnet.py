import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch

def _load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run():
    print("=== test_hrnet ===")
    base = os.path.join(os.path.dirname(__file__), "..", "hrnet")
    results = []

    try:
        mod = _load_module("hrnet_model", os.path.join(base, "model.py"))
        assert hasattr(mod, "HRNetLandmarkModel")
        print("  [PASS] HRNetLandmarkModel import")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] HRNetLandmarkModel import: {e}")
        results.append(False)

    try:
        mod = _load_module("hrnet_dataset", os.path.join(base, "dataset.py"))
        assert hasattr(mod, "LizardDataset")
        print("  [PASS] LizardDataset import")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] LizardDataset import: {e}")
        results.append(False)

    try:
        mod = _load_module("hrnet_model2", os.path.join(base, "model.py"))
        HRNetLandmarkModel = mod.HRNetLandmarkModel
        model = HRNetLandmarkModel(pretrained=False)
        model.eval()
        x = torch.rand(1, 3, 512, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 9, 2), f"Expected (1, 9, 2), got {out.shape}"
        print("  [PASS] HRNetLandmarkModel forward pass shape (1, 9, 2)")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] HRNetLandmarkModel forward pass: {e}")
        results.append(False)

    try:
        config_path = os.path.join(base, "configs", "default.json")
        with open(config_path) as f:
            config = json.load(f)
        required = ["num_landmarks", "input_size", "lr_backbone", "lr_head", "weight_decay",
                    "batch_size", "epochs", "val_fraction", "num_workers",
                    "scheduler_factor", "scheduler_patience", "grad_clip_max_norm"]
        for key in required:
            assert key in config, f"Missing key: {key}"
        print("  [PASS] hrnet config keys present")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] hrnet config keys: {e}")
        results.append(False)

    passed = sum(results)
    failed = len(results) - passed
    return passed, failed

if __name__ == "__main__":
    p, f = run()
    print(f"  {p} passed, {f} failed")
