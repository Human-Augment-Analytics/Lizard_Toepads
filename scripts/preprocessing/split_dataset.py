import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple
import yaml


def is_image_file(path: Path, allowed_exts: Iterable[str]) -> bool:
    return path.suffix.lower() in allowed_exts


def find_image_label_pairs(
    images_dir: Path,
    labels_dir: Path,
    allowed_exts: Iterable[str],
) -> List[Tuple[Path, Path]]:
    """
    Discover image files and their corresponding YOLO label files by matching stems.
    Only pairs where both image and label exist are returned.
    """
    pairs: List[Tuple[Path, Path]] = []
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    images = [p for p in images_dir.iterdir() if p.is_file() and is_image_file(p, allowed_exts)]
    images.sort()

    for img in images:
        label = labels_dir / f"{img.stem}.txt"
        if label.exists():
            pairs.append((img, label))

    return pairs


def group_pairs_by_base_name(pairs: List[Tuple[Path, Path]], suffix: str = "_flipud") -> List[List[Tuple[Path, Path]]]:
    """
    Group augmented images with their base images to avoid data leakage.

    For example, if suffix="_flipud", then:
    - "001.jpg" and "001_flipud.jpg" will be grouped together
    - Both will be assigned to the same split (train or val)

    Args:
        pairs: List of (image_path, label_path) tuples
        suffix: Suffix used for augmented images (e.g., "_flipud")

    Returns:
        List of groups, where each group is a list of (image, label) pairs
    """
    from collections import defaultdict
    groups_dict = defaultdict(list)

    for img, lbl in pairs:
        # Remove suffix to get base name
        base_name = img.stem.replace(suffix, "")
        groups_dict[base_name].append((img, lbl))

    # Convert dict to list of groups
    groups = list(groups_dict.values())
    return groups


def make_link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Replace existing file/link to ensure idempotency
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if copy:
        shutil.copy2(src, dst)
    else:
        # Use relative symlink for portability within repo
        try:
            rel_src = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel_src)
        except OSError:
            # Fallback to absolute symlink if relative fails
            dst.symlink_to(src)


def write_split_lists(output_dir: Path, train_imgs: List[Path], val_imgs: List[Path]) -> None:
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_dir / "train.txt", "w") as f:
        for p in train_imgs:
            f.write(str(p.resolve()) + "\n")
    with open(splits_dir / "val.txt", "w") as f:
        for p in val_imgs:
            f.write(str(p.resolve()) + "\n")


def split_and_materialize(
    images_dir: Path,
    labels_dir: Path,
    output_root: Path,
    train_ratio: float,
    seed: int,
    copy: bool,
    allowed_exts: Iterable[str],
    group_by_suffix: str = None,
) -> Tuple[int, int]:
    pairs = find_image_label_pairs(images_dir, labels_dir, allowed_exts)
    if not pairs:
        raise RuntimeError("No (image, label) pairs found. Ensure labels/*.txt match images by stem.")

    rng = random.Random(seed)

    # If group_by_suffix is specified, group augmented images with their originals
    if group_by_suffix:
        print(f"Grouping images by suffix '{group_by_suffix}' to avoid data leakage...")
        groups = group_pairs_by_base_name(pairs, group_by_suffix)
        print(f"Found {len(groups)} base image groups (total {len(pairs)} images)")

        # Shuffle groups instead of individual pairs
        rng.shuffle(groups)

        # Split groups
        num_train_groups = int(len(groups) * train_ratio)
        train_groups = groups[:num_train_groups]
        val_groups = groups[num_train_groups:]

        # Flatten groups back to pairs
        train_pairs = [pair for group in train_groups for pair in group]
        val_pairs = [pair for group in val_groups for pair in group]

        print(f"Train groups: {len(train_groups)} ({len(train_pairs)} images)")
        print(f"Val groups: {len(val_groups)} ({len(val_pairs)} images)")
    else:
        # Original behavior: shuffle individual pairs
        rng.shuffle(pairs)
        num_train = int(len(pairs) * train_ratio)
        train_pairs = pairs[:num_train]
        val_pairs = pairs[num_train:]

    images_train_dir = output_root / "images" / "train"
    images_val_dir = output_root / "images" / "val"
    labels_train_dir = output_root / "labels" / "train"
    labels_val_dir = output_root / "labels" / "val"

    for img, lbl in train_pairs:
        make_link_or_copy(img, images_train_dir / f"{img.name}", copy)
        make_link_or_copy(lbl, labels_train_dir / f"{lbl.name}", copy)
    for img, lbl in val_pairs:
        make_link_or_copy(img, images_val_dir / f"{img.name}", copy)
        make_link_or_copy(lbl, labels_val_dir / f"{lbl.name}", copy)

    # Write lists for convenience
    write_split_lists(output_root, [p for p, _ in train_pairs], [p for p, _ in val_pairs])

    return len(train_pairs), len(val_pairs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split YOLO dataset from flat processed dirs into train/val folders. "
            "By default uses symlinks (no extra storage)."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/H1.yaml",
        help="Path to YAML config file (default: configs/project.yaml). CLI flags override config.",
    )
    parser.add_argument(
        "--images-dir",
        help="Directory containing processed images",
    )
    parser.add_argument(
        "--labels-dir",
        help="Directory containing YOLO label .txt files",
    )
    parser.add_argument(
        "--output-dir",
        help="Output dataset root (creates images/{train,val} and labels/{train,val})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        help="Proportion of data used for training (0–1)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for split")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks",
    )
    parser.add_argument(
        "--exts",
        help="Comma-separated list of allowed image extensions",
    )
    parser.add_argument(
        "--group-by-suffix",
        help="Group images with this suffix together (e.g., '_flipud') to avoid data leakage",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load YAML config if provided
    cfg = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    def get_opt(name: str, default):
        # CLI has highest precedence
        value = getattr(args, name.replace("-", "_"), None)
        if value not in (None, ""):
            return value
        # then config YAML
        # support nested under 'split' section
        if name in cfg and cfg[name] not in (None, ""):
            return cfg[name]
        if isinstance(cfg.get("split"), dict) and cfg["split"].get(name) not in (None, ""):
            return cfg["split"][name]
        return default

    images_dir = Path(get_opt("images-dir", "data/processed/images"))
    labels_dir = Path(get_opt("labels-dir", "data/processed/labels"))
    output_root = Path(get_opt("output-dir", "data/dataset"))
    train_ratio = float(get_opt("train-ratio", 0.8))
    seed = int(get_opt("seed", 42))
    copy_flag = bool(get_opt("copy", False))
    exts_raw = str(get_opt("exts", ".jpg,.jpeg,.png"))
    allowed_exts = [e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}" for e in exts_raw.split(",") if e.strip()]
    group_by_suffix = get_opt("group-by-suffix", None)

    output_root.mkdir(parents=True, exist_ok=True)

    train_count, val_count = split_and_materialize(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_root=output_root,
        train_ratio=train_ratio,
        seed=seed,
        copy=copy_flag,
        allowed_exts=allowed_exts,
        group_by_suffix=group_by_suffix,
    )

    print("=== Split complete ===")
    print(f"Output root: {output_root}")
    print(f"Train pairs: {train_count}")
    print(f"Val pairs:   {val_count}")
    print("Hint: Update data/data.yaml to:\n"
          f"  path: {output_root.resolve()}\n"
          "  train: images/train\n"
          "  val: images/val\n")


if __name__ == "__main__":
    main()


