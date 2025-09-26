import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven YOLO training")
    parser.add_argument("--config", default="configs/project.yaml", help="Path to project YAML with train.* section (default: configs/project.yaml)")
    # Allow overriding some common options via CLI
    parser.add_argument("--data")
    parser.add_argument("--model")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device")  # e.g. 0 or 0,1 or [0,1]
    parser.add_argument("--name")
    return parser.parse_args()


def get_opt(cfg: dict, key: str, default=None, section: str = "train"):
    # Prefer top-level key, then section.key
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    if isinstance(cfg.get(section), dict) and cfg[section].get(key) is not None:
        return cfg[section][key]
    return default


def main() -> None:
    args = parse_args()
    cfg = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    # Build dataset config dict if provided in project.yaml
    dataset_cfg = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else None
    data_from_dataset = None
    if dataset_cfg:
        # Accept either absolute file lists or a path+relative subpaths
        data_from_dataset = {}
        for k in ("path", "train", "val", "test", "nc", "names"):
            if dataset_cfg.get(k) is not None:
                data_from_dataset[k] = dataset_cfg[k]

    # Merge precedence for data: CLI --data > train.data in config > dataset section dict > default
    data = args.data or get_opt(cfg, "data", None) or data_from_dataset or "data/data.yaml"
    model_path = args.model or get_opt(cfg, "model", "yolov11n.pt")
    epochs = args.epochs or get_opt(cfg, "epochs", 100)
    batch = args.batch or get_opt(cfg, "batch", 32)
    imgsz = args.imgsz or get_opt(cfg, "imgsz", 1024)
    workers = get_opt(cfg, "workers", 8)
    patience = get_opt(cfg, "patience", 20)
    name = args.name or get_opt(cfg, "name", "tps_yolo_exp")
    device = args.device or get_opt(cfg, "device", 0)
    amp = get_opt(cfg, "amp", True)
    cache = get_opt(cfg, "cache", True)
    save_period = get_opt(cfg, "save_period", 10)

    model = YOLO(model_path)
    model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        workers=workers,
        patience=patience,
        name=name,
        device=device,
        amp=amp,
        cache=cache,
        save_period=save_period,
    )


if __name__ == "__main__":
    main()