import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

def test_imports():
    from common.tps_utils import get_tps_coords
    from common.yolo_utils import crop_toe_boxes
    from common.heatmap_utils import tps_to_heatmap, generate_overlay
    print("  [PASS] common imports")

def test_tps_to_heatmap_shape():
    from common.heatmap_utils import tps_to_heatmap
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    tps = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0), (40.0, 40.0),
           (50.0, 50.0), (15.0, 15.0), (25.0, 25.0), (35.0, 35.0), (45.0, 45.0)]
    result = tps_to_heatmap(tps, crop)
    assert result.shape == (64, 64, 9), f"Expected (64, 64, 9), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    print("  [PASS] tps_to_heatmap shape and dtype")

def test_generate_overlay_shape():
    from common.heatmap_utils import tps_to_heatmap, generate_overlay
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    tps = [(10.0, 10.0)] * 9
    heatmaps = tps_to_heatmap(tps, crop)
    overlay = generate_overlay(crop, heatmaps)
    assert overlay.shape == (64, 64, 3), f"Expected (64, 64, 3), got {overlay.shape}"
    print("  [PASS] generate_overlay spatial dimensions preserved")

def test_crop_toe_boxes_returns_tuple():
    from common.yolo_utils import crop_toe_boxes

    class MockBoxes:
        def __init__(self):
            import torch
            self.xyxy = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
            self.cls = torch.tensor([2.0])
            self.conf = torch.tensor([0.9])

    class MockResult:
        def __init__(self):
            self.boxes = MockBoxes()

    r = [MockResult()]
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    g_coords = {
        "finger": [(20.0, 20.0), (30.0, 30.0), (25.0, 25.0),
                   (15.0, 15.0), (35.0, 35.0), (20.0, 30.0),
                   (30.0, 20.0), (25.0, 15.0), (25.0, 35.0)],
        "toe": []
    }
    result = crop_toe_boxes(r, image, g_coords)
    assert isinstance(result, tuple) and len(result) == 3, f"Expected 3-tuple, got {type(result)}"
    print("  [PASS] crop_toe_boxes returns 3-tuple")

def run():
    print("=== test_common ===")
    tests = [test_imports, test_tps_to_heatmap_shape, test_generate_overlay_shape, test_crop_toe_boxes_returns_tuple]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
    return passed, failed

if __name__ == "__main__":
    p, f = run()
    print(f"  {p} passed, {f} failed")
