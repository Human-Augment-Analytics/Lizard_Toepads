import argparse
from pathlib import Path
import yaml
import tempfile
import os
from ultralytics import YOLO
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven YOLO training")
    parser.add_argument("--config", default="configs/H1.yaml", help="Path to project YAML with train.* section (default: configs/H1.yaml)")
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


def create_temp_data_yaml(data_dict):
    """Create a temporary YAML file from dataset dictionary"""
    temp_dir = Path("data")
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / "temp_data.yaml"

    with open(temp_file, "w") as f:
        yaml.dump(data_dict, f, default_flow_style=False)

    return str(temp_file)


def ensure_model_exists(model_path: str) -> str:
    """Ensure model exists, download if necessary"""
    # If it's already a full path and exists, return it
    path = Path(model_path)
    if path.exists():
        return str(path)

    # Check if it's a path to models/base_models/
    if "models/base_models/" in model_path or "models\\base_models\\" in model_path:
        # Extract just the model name
        model_name = Path(model_path).name  # e.g., "yolov11n.pt"

        # Try to find in models/base_models directory
        base_models_dir = Path("models/base_models")
        base_models_dir.mkdir(parents=True, exist_ok=True)

        local_model_path = base_models_dir / model_name

        if local_model_path.exists():
            print(f"Found existing model: {local_model_path}")
            return str(local_model_path)

        # Download using just the model name (YOLO will auto-download)
        print(f"Model {model_name} not found locally. Downloading...")

        try:
            # YOLO will download when initialized with just the model name
            temp_model = YOLO(model_name)

            # Find the downloaded model (usually in Ultralytics cache)
            import shutil
            from ultralytics.utils import SETTINGS

            # Try to find the downloaded model
            possible_paths = [
                Path.home() / ".cache" / "ultralytics" / "assets" / model_name,
                Path(SETTINGS.get('weights_dir', '')) / model_name if SETTINGS.get('weights_dir') else None,
            ]

            for cache_path in possible_paths:
                if cache_path and cache_path.exists():
                    # Copy to our local directory
                    print(f"Copying model from {cache_path} to {local_model_path}")
                    shutil.copy2(cache_path, local_model_path)
                    return str(local_model_path)

            # If we can't find it in cache, just use the model name
            # YOLO will handle it
            print(f"Model downloaded but cache location unknown. Using model name directly.")
            return model_name

        except Exception as e:
            print(f"Warning during model setup: {e}")
            # Just use the model name and let YOLO handle it
            return model_name

    # For simple model names, return as is (let YOLO handle download)
    if model_path.endswith('.pt') and '/' not in model_path and '\\' not in model_path:
        print(f"Using model: {model_path}")
        return model_path

    # For other cases, return as is
    return model_path


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
    data_raw = args.data or get_opt(cfg, "data", None) or data_from_dataset or "data/data.yaml"

    # Handle dict data by creating temporary YAML file
    if isinstance(data_raw, dict):
        print(f"Creating temporary data.yaml from config...")
        data = create_temp_data_yaml(data_raw)
    else:
        data = data_raw
    model_path = args.model or get_opt(cfg, "model", "models/base_models/yolov11n.pt")
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

    # Data augmentation parameters (all optional, use YOLO defaults if not specified)
    flipud = get_opt(cfg, "flipud", None)      # Vertical flip probability (0.0-1.0)
    fliplr = get_opt(cfg, "fliplr", None)      # Horizontal flip probability (0.0-1.0)
    degrees = get_opt(cfg, "degrees", None)    # Rotation range in degrees (0-180)
    translate = get_opt(cfg, "translate", None) # Translation fraction (0.0-1.0)
    scale = get_opt(cfg, "scale", None)        # Scaling factor (>=0.0)
    shear = get_opt(cfg, "shear", None)        # Shear angle in degrees (-180 to +180)
    perspective = get_opt(cfg, "perspective", None)  # Perspective transform (0.0-0.001)
    hsv_h = get_opt(cfg, "hsv_h", None)        # Hue variation (0.0-1.0)
    hsv_s = get_opt(cfg, "hsv_s", None)        # Saturation variation (0.0-1.0)
    hsv_v = get_opt(cfg, "hsv_v", None)        # Value/brightness variation (0.0-1.0)
    mosaic = get_opt(cfg, "mosaic", None)      # Mosaic augmentation probability (0.0-1.0)
    mixup = get_opt(cfg, "mixup", None)        # Mixup augmentation probability (0.0-1.0)
    copy_paste = get_opt(cfg, "copy_paste", None) # Copy-paste augmentation probability (0.0-1.0)
    cos_lr = get_opt(cfg, "cos_lr", None)      # Cosine learning rate scheduler (bool)
    close_mosaic = get_opt(cfg, "close_mosaic", None) # Disable mosaic for last N epochs (int)

    # Ensure model exists or download it
    model_path = ensure_model_exists(model_path)
    print(f"Using model: {model_path}")

    # Build training arguments dict (only include non-None augmentation params)
    train_args = {
        "data": data,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "workers": workers,
        "patience": patience,
        "name": name,
        "device": device,
        "amp": amp,
        "cache": cache,
        "save_period": save_period,
    }

    # Add augmentation parameters if specified in config
    if flipud is not None:
        train_args["flipud"] = flipud
    if fliplr is not None:
        train_args["fliplr"] = fliplr
    if degrees is not None:
        train_args["degrees"] = degrees
    if translate is not None:
        train_args["translate"] = translate
    if scale is not None:
        train_args["scale"] = scale
    if shear is not None:
        train_args["shear"] = shear
    if perspective is not None:
        train_args["perspective"] = perspective
    if hsv_h is not None:
        train_args["hsv_h"] = hsv_h
    if hsv_s is not None:
        train_args["hsv_s"] = hsv_s
    if hsv_v is not None:
        train_args["hsv_v"] = hsv_v
    if mosaic is not None:
        train_args["mosaic"] = mosaic
    if mixup is not None:
        train_args["mixup"] = mixup
    if copy_paste is not None:
        train_args["copy_paste"] = copy_paste
    if cos_lr is not None:
        train_args["cos_lr"] = cos_lr
    if close_mosaic is not None:
        train_args["close_mosaic"] = close_mosaic

    # Print augmentation configuration
    aug_params = {k: v for k, v in train_args.items() if k in [
        "flipud", "fliplr", "degrees", "translate", "scale", "shear",
        "perspective", "hsv_h", "hsv_s", "hsv_v", "mosaic",
        "mixup", "copy_paste", "cos_lr", "close_mosaic"
    ]}
    if aug_params:
        print("\n=== Data Augmentation Configuration ===")
        for k, v in aug_params.items():
            print(f"  {k}: {v}")
        print("=" * 40 + "\n")
    else:
        print("\n=== Using YOLO default augmentation settings ===\n")

    model = YOLO(model_path)
    model.train(**train_args)

    # Explicitly run validation to ensure plots (confusion matrix, results.png) are generated
    print("\nTraining complete. Running final validation to generate plots...")
    model.val(data=data, plots=True)


if __name__ == "__main__":
    main()