import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stacked-hourglass"))

import json
import torch

def test_model_import():
    from stacked_hourglass_model import StackedHourGlass
    print("  [PASS] StackedHourGlass import")

def test_dataset_import():
    from stacked_hourglass_dataset import LizardDataset, apply_base_transform, apply_augmentation
    print("  [PASS] LizardDataset, apply_base_transform, apply_augmentation import")

def test_model_forward():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stacked-hourglass"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "shg_model",
        os.path.join(os.path.dirname(__file__), "..", "stacked-hourglass", "model.py")
    )
    mod = importlib.util.load_from_spec(spec)
    spec.loader.exec_module(mod)
    StackedHourGlass = mod.StackedHourGlass

    model = StackedHourGlass()
    model.eval()
    x = torch.rand(1, 512, 512, 3)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2, 9, 128, 128), f"Expected (1, 2, 9, 128, 128), got {out.shape}"
    print("  [PASS] StackedHourGlass forward pass shape (1, 2, 9, 128, 128)")

def test_config_keys():
    config_path = os.path.join(os.path.dirname(__file__), "..", "stacked-hourglass", "configs", "default.json")
    with open(config_path) as f:
        config = json.load(f)
    required = ["augmentationFactor", "epochs", "trainTestSplit", "initialLR", "scheduler", "batchSize", "randomState"]
    for key in required:
        assert key in config, f"Missing key: {key}"
    print("  [PASS] stacked-hourglass config keys present")

def _load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run():
    print("=== test_stacked_hourglass ===")
    base = os.path.join(os.path.dirname(__file__), "..", "stacked-hourglass")

    tests_results = []

    try:
        mod = _load_module("shg_model", os.path.join(base, "model.py"))
        StackedHourGlass = mod.StackedHourGlass
        print("  [PASS] StackedHourGlass import")
        tests_results.append(True)
    except Exception as e:
        print(f"  [FAIL] StackedHourGlass import: {e}")
        tests_results.append(False)

    try:
        mod = _load_module("shg_dataset", os.path.join(base, "dataset.py"))
        assert hasattr(mod, "LizardDataset")
        assert hasattr(mod, "apply_base_transform")
        assert hasattr(mod, "apply_augmentation")
        print("  [PASS] dataset imports")
        tests_results.append(True)
    except Exception as e:
        print(f"  [FAIL] dataset imports: {e}")
        tests_results.append(False)

    try:
        mod = _load_module("shg_model2", os.path.join(base, "model.py"))
        StackedHourGlass = mod.StackedHourGlass
        model = StackedHourGlass()
        model.eval()
        x = torch.rand(1, 512, 512, 3)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 9, 128, 128), f"Expected (1, 2, 9, 128, 128), got {out.shape}"
        print("  [PASS] StackedHourGlass forward pass shape (1, 2, 9, 128, 128)")
        tests_results.append(True)
    except Exception as e:
        print(f"  [FAIL] StackedHourGlass forward pass: {e}")
        tests_results.append(False)

    try:
        config_path = os.path.join(base, "configs", "default.json")
        with open(config_path) as f:
            config = json.load(f)
        required = ["augmentationFactor", "epochs", "trainTestSplit", "initialLR", "scheduler", "batchSize", "randomState"]
        for key in required:
            assert key in config, f"Missing key: {key}"
        print("  [PASS] config keys present")
        tests_results.append(True)
    except Exception as e:
        print(f"  [FAIL] config keys: {e}")
        tests_results.append(False)

    passed = sum(tests_results)
    failed = len(tests_results) - passed
    return passed, failed

if __name__ == "__main__":
    p, f = run()
    print(f"  {p} passed, {f} failed")
