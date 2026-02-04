import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import ConcatDataset, DataLoader, random_split


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __package__ in (None, ""):
    base_dir = Path(__file__).resolve().parent
    config_module = _load_module("_pytorch_keypoint_config", base_dir / "config.py")
    dataset_module = _load_module("_pytorch_keypoint_dataset", base_dir / "dataset.py")
    model_module = _load_module("_pytorch_keypoint_model", base_dir / "model.py")

    load_config = config_module.load_config
    ToepadKeypointDataset = dataset_module.ToepadKeypointDataset
    ModelConfig = model_module.ModelConfig
    KeypointRegressor = model_module.KeypointRegressor
else:
    from .config import load_config
    from .dataset import ToepadKeypointDataset
    from .model import KeypointRegressor, ModelConfig


def build_datasets(cfg):
    train_dataset = ToepadKeypointDataset(
        xml_path=str(cfg.data.train_xml),
        root_dir=str(cfg.data.root),
        image_size=cfg.data.image_size,
        augment=cfg.data.augment,
    )
    val_dataset = ToepadKeypointDataset(
        xml_path=str(cfg.data.val_xml),
        root_dir=str(cfg.data.root),
        image_size=cfg.data.image_size,
        augment=False,
    )
    return train_dataset, val_dataset


def build_dataloaders(cfg, train_dataset, val_dataset):
    seed = pl.seed_everything(42)
    generator = torch.Generator()
    generator.manual_seed(seed if isinstance(seed, int) else 42)
    val_length = int(0.1 * len(train_dataset))
    if val_length > 0:
        train_subset, extra_val_subset = random_split(
            train_dataset,
            [len(train_dataset) - val_length, val_length],
            generator=generator,
        )
        combined_val = ConcatDataset([val_dataset, extra_val_subset])
    else:
        train_subset = train_dataset
        combined_val = val_dataset

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.trainer.pin_memory and torch.cuda.is_available(),
        persistent_workers=cfg.trainer.persistent_workers and cfg.data.num_workers > 0,
    )

    val_loader = DataLoader(
        combined_val,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.trainer.pin_memory and torch.cuda.is_available(),
        persistent_workers=cfg.trainer.persistent_workers and cfg.data.num_workers > 0,
    )
    return train_loader, val_loader


def log_runtime_device(trainer: pl.Trainer) -> None:
    """Print the device type that Lightning will use for training."""
    root_device = getattr(trainer.strategy, "root_device", torch.device("cpu"))
    if root_device.type == "cuda":
        index = root_device.index if root_device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        print(f"Training on GPU [{index}]: {name}")
    elif root_device.type == "mps":
        print("Training on Apple MPS accelerator.")
    else:
        print("Training on CPU.")


def main():
    parser = argparse.ArgumentParser(description="Train GPU-accelerated keypoint detector for lizard toepads.")
    parser.add_argument("--config", default="configs/keypoints.yaml", help="Path to YAML config file.")
    parser.add_argument("--max-epochs", type=int, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--precision", type=str, help="Override precision mode (e.g., 16-mixed, 32).")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.max_epochs:
        cfg.trainer.max_epochs = args.max_epochs
    if args.batch_size:
        cfg.trainer.batch_size = args.batch_size
    if args.precision:
        cfg.trainer.precision = args.precision

    train_dataset, val_dataset = build_datasets(cfg)
    num_keypoints = train_dataset.num_keypoints

    train_loader, val_loader = build_dataloaders(cfg, train_dataset, val_dataset)

    model_cfg = ModelConfig(num_keypoints=num_keypoints, **cfg.model)
    model = KeypointRegressor(model_cfg)

    accelerator = cfg.trainer.accelerator
    devices = cfg.trainer.devices
    precision = cfg.trainer.precision

    if accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
            if devices == "auto":
                devices = 1
            print(f"Detected CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            torch.set_float32_matmul_precision("high")
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            if devices == "auto":
                devices = 1
            print("Detected Apple MPS accelerator.")
        else:
            accelerator = "cpu"
            if precision.endswith("mixed"):
                precision = "32"
    elif accelerator == "gpu" and not torch.cuda.is_available():
        print("Requested GPU accelerator but no CUDA device found; falling back to CPU.")
        accelerator = "cpu"
        if precision.endswith("mixed"):
            precision = "32"

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            filename="keypoints-{epoch:02d}-{val_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=precision,
        default_root_dir=str(cfg.trainer.default_root_dir),
        log_every_n_steps=10,
        callbacks=callbacks,
    )

    log_runtime_device(trainer)

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

