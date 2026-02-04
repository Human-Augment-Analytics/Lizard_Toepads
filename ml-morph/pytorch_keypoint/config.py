from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataConfig:
    root: Path
    train_xml: Path
    val_xml: Path
    image_size: int = 256
    num_workers: int = 4
    augment: bool = False


@dataclass
class TrainerConfig:
    batch_size: int = 16
    max_epochs: int = 50
    accelerator: str = "auto"
    devices: Any = "auto"
    precision: str = "16-mixed"
    default_root_dir: Path = Path("runs/keypoints")
    pin_memory: bool = True
    persistent_workers: bool = False


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: Dict[str, Any]
    trainer: TrainerConfig


def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_cfg = raw.get("data", {})
    trainer_cfg = raw.get("trainer", {})
    model_cfg = raw.get("model", {})

    data = DataConfig(
        root=Path(data_cfg.get("root", "ml-morph")),
        train_xml=Path(data_cfg.get("train_xml", "ml-morph/train.xml")),
        val_xml=Path(data_cfg.get("val_xml", "ml-morph/test.xml")),
        image_size=int(data_cfg.get("image_size", 256)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        augment=bool(data_cfg.get("augment", False)),
    )

    trainer = TrainerConfig(
        batch_size=int(trainer_cfg.get("batch_size", 16)),
        max_epochs=int(trainer_cfg.get("max_epochs", 50)),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", "auto"),
        precision=trainer_cfg.get("precision", "16-mixed"),
        default_root_dir=Path(trainer_cfg.get("default_root_dir", "runs/keypoints")),
        pin_memory=bool(trainer_cfg.get("pin_memory", True)),
        persistent_workers=bool(trainer_cfg.get("persistent_workers", False)),
    )

    return ExperimentConfig(
        data=data,
        model=model_cfg,
        trainer=trainer,
    )

