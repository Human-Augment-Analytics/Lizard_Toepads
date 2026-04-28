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
    print("=== test_vit ===")
    base = os.path.join(os.path.dirname(__file__), "..", "vit")
    results = []

    try:
        mod = _load_module("vit_model", os.path.join(base, "model.py"))
        assert hasattr(mod, "ViTLandmark")
        print("  [PASS] ViTLandmark import")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] ViTLandmark import: {e}")
        results.append(False)

    try:
        mod = _load_module("vit_dataset", os.path.join(base, "dataset.py"))
        assert hasattr(mod, "ViTDataset")
        print("  [PASS] ViTDataset import")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] ViTDataset import: {e}")
        results.append(False)

    try:
        mod = _load_module("vit_model2", os.path.join(base, "model.py"))
        ViTLandmark = mod.ViTLandmark
        model = ViTLandmark(pretrained=False)
        model.eval()
        x = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 18), f"Expected (1, 18), got {out.shape}"
        print("  [PASS] ViTLandmark forward pass shape (1, 18)")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] ViTLandmark forward pass: {e}")
        results.append(False)

    try:
        config_path = os.path.join(base, "configs", "default.json")
        with open(config_path) as f:
            config = json.load(f)
        required = ["num_landmarks", "backbone_lr", "head_lr", "momentum", "batch_size", "epochs", "val_fraction", "grad_clip_max_norm"]
        for key in required:
            assert key in config, f"Missing key: {key}"
        print("  [PASS] vit config keys present")
        results.append(True)
    except Exception as e:
        print(f"  [FAIL] vit config keys: {e}")
        results.append(False)

    passed = sum(results)
    failed = len(results) - passed
    return passed, failed

if __name__ == "__main__":
    p, f = run()
    print(f"  {p} passed, {f} failed")
