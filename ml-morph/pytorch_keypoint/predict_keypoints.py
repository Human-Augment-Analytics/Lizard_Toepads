import argparse
import csv
import importlib.util
import pickle
from pathlib import Path
import sys
from typing import List

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

try:
    from .config import load_config
    from .dataset import ToepadKeypointDataset
    from .model import KeypointRegressor, ModelConfig
except ImportError:
    module_root = Path(__file__).resolve().parent

    def _load_module(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import module {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    config_module = _load_module("_pytorch_keypoint_config", module_root / "config.py")
    dataset_module = _load_module("_pytorch_keypoint_dataset", module_root / "dataset.py")
    model_module = _load_module("_pytorch_keypoint_model", module_root / "model.py")

    load_config = config_module.load_config  # type: ignore[attr-defined]
    ToepadKeypointDataset = dataset_module.ToepadKeypointDataset  # type: ignore[attr-defined]
    ModelConfig = model_module.ModelConfig  # type: ignore[attr-defined]
    KeypointRegressor = model_module.KeypointRegressor  # type: ignore[attr-defined]


def _ensure_list(item):
    if isinstance(item, list):
        return item
    return [item]


def _resolve_checkpoint(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        ckpts = sorted(candidate.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint files found in {candidate}")
        return ckpts[-1]
    raise FileNotFoundError(f"Checkpoint path not found: {path}")


def main():
    parser = argparse.ArgumentParser(description="Run keypoint predictions using a trained checkpoint.")
    parser.add_argument("--config", default="configs/keypoints.yaml", help="Path to experiment config.")
    parser.add_argument("--checkpoint", required=True, help="Path to Lightning checkpoint (.ckpt) or directory.")
    parser.add_argument(
        "--xml",
        default=None,
        help="Annotation XML to run predictions on (defaults to config data.val_xml).",
    )
    parser.add_argument("--output", default="predictions/keypoints_test.csv", help="Path to output CSV.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override number of DataLoader workers.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (for debugging).")

    args = parser.parse_args()

    cfg = load_config(args.config)
    xml_path = Path(args.xml or cfg.data.val_xml)
    if not xml_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {xml_path}")

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = ToepadKeypointDataset(
        xml_path=str(xml_path),
        root_dir=str(cfg.data.root),
        image_size=cfg.data.image_size,
        augment=False,
        return_metadata=True,
    )

    num_keypoints = dataset.num_keypoints

    if args.limit is not None:
        indices = list(range(min(args.limit, len(dataset))))
        dataset = Subset(dataset, indices)

    num_workers = args.num_workers if args.num_workers is not None else cfg.data.num_workers
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_cfg = ModelConfig(num_keypoints=num_keypoints, **cfg.model)
    model = KeypointRegressor(model_cfg)
    try:
        state = torch.load(checkpoint_path, map_location="cpu")
    except pickle.UnpicklingError:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    model.load_state_dict(state_dict)
    model.eval()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    records: List[List[float]] = []
    headers = ["image"]
    num_keypoints = model_cfg.num_keypoints
    for k in range(num_keypoints):
        headers.append(f"k{k}_x")
        headers.append(f"k{k}_y")

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            preds = model(images).cpu().view(images.size(0), num_keypoints, 2)
            metadata = batch["metadata"]

            image_paths = _ensure_list(metadata["image_path"])
            bboxes = metadata["bbox"]
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.cpu().numpy()
            else:
                if (
                    isinstance(bboxes, (list, tuple))
                    and len(bboxes) == 4
                    and all(isinstance(x, torch.Tensor) for x in bboxes)
                ):
                    bboxes = torch.stack(list(bboxes), dim=1).cpu().numpy()
                else:
                    bboxes = np.asarray(bboxes, dtype=np.float32)
                    if bboxes.ndim == 1:
                        bboxes = bboxes[None, :]

            for i in range(images.size(0)):
                bbox = np.asarray(bboxes[i], dtype=np.float32).flatten()
                if bbox.size < 4:
                    raise ValueError(f"Expected bbox with 4 values, got {bbox}")
                left, top, width, height = bbox[:4]
                kp_abs = preds[i].clone()
                kp_abs[:, 0] = kp_abs[:, 0] * width + left
                kp_abs[:, 1] = kp_abs[:, 1] * height + top

                row: List[float] = [image_paths[i]]
                for point in kp_abs.tolist():
                    row.extend(point)
                records.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(records)

    print(f"Saved {len(records)} predictions to {output_path}")


if __name__ == "__main__":
    main()

