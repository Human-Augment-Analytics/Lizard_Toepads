"""Cephalometric (ISBI 2015) dataset preprocessor.

Converts raw ISBI lateral skull radiographs and the two annotator coordinate
files (senior / junior) into .pt files compatible with CephalometricDataset.

The ISBI 2015 dataset provides 400 lateral cephalograms with exactly 19
anatomical landmarks per image, annotated independently by a senior and a
junior orthodontist. The ground-truth position for each landmark is the
per-landmark average of the two annotators.

Each output .pt file contains:
  - "image":         (3, canvas, canvas) uint8 tensor — grayscale replicated to 3 channels
  - "tps":           (19, 2) float32 tensor — landmark coordinates normalized to [0, 1]
  - "orig_size":     (2,) float32 tensor — [H, W] of the source radiograph
  - "pixel_spacing": scalar float32 tensor — mm per pixel (ISBI default 0.1)
  - "split":         str — "train" | "test1" | "test2"

Split assignment follows the standard ISBI protocol by filename number:
  train  = images 1-150
  test1  = images 151-300
  test2  = images 301-400

Usage:
    python -m landmarking.datasets.cephalometric.preprocess \\
        --image-root path/to/RawImage/ \\
        --senior-annotations path/to/400_senior/ \\
        --junior-annotations path/to/400_junior/ \\
        --output-dir path/to/Cephalometric_data/ \\
        --pixel-spacing 0.1
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

NUM_LANDMARKS = 19

# Image filename number -> split. train 1-150, test1 151-300, test2 301-400.
TRAIN_MAX = 150
TEST1_MAX = 300
TEST2_MAX = 400


# ── Pure, importable helpers (no cv2 / image IO required) ────────────────────


def parse_annotation_file(path) -> np.ndarray:
    """Parse an ISBI annotator file into a (19, 2) float array.

    The canonical ISBI format is one landmark per line, ``x,y`` (comma
    separated), for exactly 19 landmark lines in standard landmark order.
    Whitespace-separated ``x y`` is also accepted defensively. Blank lines are
    ignored, but any non-empty line that yields other than two numeric values
    is treated as a parse failure.

    Args:
        path: Path to the annotator coordinate file.

    Returns:
        A (19, 2) float64 ndarray of pixel (x, y) coordinates.

    Raises:
        ValueError: If the file cannot be parsed or does not yield exactly 19
            (x, y) landmark rows. The message names the offending file.
    """
    path = Path(path)
    try:
        with open(path, "r") as f:
            raw_lines = f.readlines()
    except OSError as e:
        raise ValueError(f"Could not read annotation file {path}: {e}") from e

    coords = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Accept "x,y" (canonical) or whitespace-separated "x y".
        if "," in stripped:
            parts = [p for p in stripped.split(",") if p.strip() != ""]
        else:
            parts = stripped.split()
        if len(parts) != 2:
            # A non-coordinate line: stop only if we already have 19 rows,
            # otherwise it is a genuine parse failure.
            if len(coords) == NUM_LANDMARKS:
                break
            raise ValueError(
                f"Annotation file {path} contains a line that is not a valid "
                f"'x,y' landmark: {stripped!r}"
            )
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError as e:
            raise ValueError(
                f"Annotation file {path} contains a non-numeric coordinate: "
                f"{stripped!r}"
            ) from e
        coords.append((x, y))

    if len(coords) != NUM_LANDMARKS:
        raise ValueError(
            f"Annotation file {path} did not yield exactly {NUM_LANDMARKS} "
            f"(x, y) landmark rows (got {len(coords)})"
        )

    return np.asarray(coords, dtype=np.float64)


def average_annotators(senior_xy: np.ndarray, junior_xy: np.ndarray) -> np.ndarray:
    """Return the per-landmark element-wise mean of two (19, 2) arrays."""
    senior_xy = np.asarray(senior_xy, dtype=np.float64)
    junior_xy = np.asarray(junior_xy, dtype=np.float64)
    if senior_xy.shape != junior_xy.shape:
        raise ValueError(
            f"Annotator coordinate shapes differ: {senior_xy.shape} vs "
            f"{junior_xy.shape}"
        )
    return (senior_xy + junior_xy) / 2.0


def to_three_channel(gray_hw: np.ndarray) -> np.ndarray:
    """Replicate a single-channel (H, W) uint8 image into (3, H, W) uint8.

    All three output channels are identical to the source channel.
    """
    gray_hw = np.asarray(gray_hw)
    if gray_hw.ndim != 2:
        raise ValueError(
            f"to_three_channel expects a 2-D (H, W) array, got shape "
            f"{gray_hw.shape}"
        )
    gray = gray_hw.astype(np.uint8, copy=False)
    return np.stack([gray, gray, gray], axis=0)


def normalize_coords(coords_xy: np.ndarray, width: float, height: float) -> np.ndarray:
    """Normalize (x, y) pixel coordinates to [0, 1] by width / height."""
    coords_xy = np.asarray(coords_xy, dtype=np.float64)
    out = coords_xy.copy()
    out[:, 0] = coords_xy[:, 0] / float(width)
    out[:, 1] = coords_xy[:, 1] / float(height)
    return out


def assign_split(image_number: int) -> str:
    """Assign the ISBI split for a 1-based image number."""
    if 1 <= image_number <= TRAIN_MAX:
        return "train"
    if TRAIN_MAX < image_number <= TEST1_MAX:
        return "test1"
    if TEST1_MAX < image_number <= TEST2_MAX:
        return "test2"
    raise ValueError(f"Image number {image_number} is outside the ISBI 1-400 range")


def _image_number_from_stem(stem: str) -> int:
    """Extract the integer image number from a filename stem (e.g. '0157')."""
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not extract an image number from stem {stem!r}")
    return int(digits)


# ── Main preprocessing driver (requires cv2 for image IO) ────────────────────


def preprocess_cephalometric(
    image_root: str,
    senior_annotations: str,
    junior_annotations: str,
    output_dir: str,
    pixel_spacing: float = 0.1,
    target_size: int = 512,
) -> None:
    """Convert raw ISBI images + annotator files into per-image .pt files.

    Args:
        image_root: Directory containing the raw radiograph images.
        senior_annotations: Directory containing the senior annotator .txt files.
        junior_annotations: Directory containing the junior annotator .txt files.
        output_dir: Root output directory; .pt files are written into
            ``train/``, ``test1/``, and ``test2/`` subdirectories.
        pixel_spacing: Physical size of one pixel in millimeters.
        target_size: Square canvas size the radiographs are resized to.
    """
    import cv2  # local import so helper-only imports never require cv2

    image_root = Path(image_root)
    senior_dir = Path(senior_annotations)
    junior_dir = Path(junior_annotations)
    output_dir = Path(output_dir)

    for split in ("train", "test1", "test2"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p
        for p in image_root.iterdir()
        if p.is_file() and p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    logging.info(
        f"Processing {len(image_paths)} radiographs from {image_root} → {output_dir}"
    )

    saved = 0
    skipped = 0

    for img_path in image_paths:
        stem = img_path.stem
        try:
            image_number = _image_number_from_stem(stem)
            split = assign_split(image_number)
        except ValueError as e:
            logging.warning(f"{img_path.name}: {e}")
            skipped += 1
            continue

        senior_file = senior_dir / f"{stem}.txt"
        junior_file = junior_dir / f"{stem}.txt"
        if not senior_file.exists() or not junior_file.exists():
            logging.warning(
                f"{img_path.name}: missing annotator file(s) "
                f"({senior_file.name} / {junior_file.name})"
            )
            skipped += 1
            continue

        # parse_annotation_file raises ValueError naming the file on bad input.
        senior_xy = parse_annotation_file(senior_file)
        junior_xy = parse_annotation_file(junior_file)
        gt_xy = average_annotators(senior_xy, junior_xy)

        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            logging.warning(f"{img_path.name}: cv2 failed to read image")
            skipped += 1
            continue

        orig_h, orig_w = gray.shape[:2]

        # Normalize coordinates against the ORIGINAL image size (Req 2.3, 3.1).
        coords_norm = normalize_coords(gt_xy, width=orig_w, height=orig_h)

        # Resize the radiograph to the square canvas and replicate to 3 channels.
        resized = cv2.resize(
            gray, (target_size, target_size), interpolation=cv2.INTER_LINEAR
        )
        img_chw = to_three_channel(resized)

        pt_data = {
            "image": torch.from_numpy(img_chw),
            "tps": torch.from_numpy(coords_norm.astype(np.float32)),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
            "pixel_spacing": torch.tensor(float(pixel_spacing), dtype=torch.float32),
            "split": split,
        }

        out_path = output_dir / split / f"{stem}.pt"
        torch.save(pt_data, str(out_path))
        saved += 1

        if saved % 50 == 0:
            logging.info(f"  {saved}/{len(image_paths)} saved...")

    logging.info(
        f"Done. Saved: {saved}, Skipped: {skipped}, Total: {len(image_paths)}"
    )


# ── Standard ISBI 2015 distribution layout driver ───────────────────────────

# Maps the split name to (image subfolder, output subdir) for the standard
# ISBI 2015 release. Image folders are pre-split, so the split is determined
# by which folder a file lives in — not by parsing its filename number.
ISBI_SPLIT_FOLDERS = {
    "train": "TrainingData",
    "test1": "Test1Data",
    "test2": "Test2Data",
}


def preprocess_cephalometric_isbi(
    data_root: str,
    output_dir: str = None,
    senior_subdir: str = "AnnotationsByMD/400_senior",
    junior_subdir: str = "AnnotationsByMD/400_junior",
    raw_image_subdir: str = "RawImage",
    pixel_spacing: float = 0.1,
    target_size: int = 512,
) -> None:
    """Preprocess the standard ISBI 2015 distribution layout.

    Expects the standard release structure under ``data_root``::

        <data_root>/
            RawImage/ (or the split folders directly under data_root)
                TrainingData/  Test1Data/  Test2Data/   (.bmp)
            AnnotationsByMD/
                400_senior/    400_junior/              (.txt)

    The split is assigned by which image folder each file lives in, so no
    filename-number parsing is required. Outputs one ``.pt`` per image into
    ``<output_dir>/{train,test1,test2}`` (defaults to ``data_root``).

    Args:
        data_root: Root of the ISBI distribution.
        output_dir: Where to write train/test1/test2 .pt dirs (default data_root).
        senior_subdir: Senior annotations dir relative to data_root.
        junior_subdir: Junior annotations dir relative to data_root.
        raw_image_subdir: Image parent dir relative to data_root; the split
            folders are searched here first, then directly under data_root.
        pixel_spacing: mm per pixel (ISBI default 0.1).
        target_size: Square canvas size for the stored image.
    """
    import cv2  # local import so helper-only imports never require cv2

    data_root = Path(data_root)
    output_dir = Path(output_dir) if output_dir else data_root
    senior_dir = data_root / senior_subdir
    junior_dir = data_root / junior_subdir

    if not senior_dir.exists():
        raise FileNotFoundError(f"Senior annotations dir not found: {senior_dir}")
    if not junior_dir.exists():
        raise FileNotFoundError(f"Junior annotations dir not found: {junior_dir}")

    for split in ("train", "test1", "test2"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    total_saved = 0
    total_skipped = 0
    img_exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    for split, folder_name in ISBI_SPLIT_FOLDERS.items():
        # Locate the image folder: try <data_root>/<raw_image_subdir>/<folder>
        # then <data_root>/<folder>.
        candidates = [
            data_root / raw_image_subdir / folder_name,
            data_root / folder_name,
        ]
        img_dir = next((c for c in candidates if c.exists()), None)
        if img_dir is None:
            logging.warning(
                f"[{split}] image folder not found in "
                f"{[str(c) for c in candidates]} — skipping split"
            )
            continue

        image_paths = sorted(
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in img_exts
        )
        logging.info(
            f"[{split}] processing {len(image_paths)} images from {img_dir}"
        )

        for img_path in image_paths:
            stem = img_path.stem
            senior_file = senior_dir / f"{stem}.txt"
            junior_file = junior_dir / f"{stem}.txt"
            if not senior_file.exists() or not junior_file.exists():
                logging.warning(
                    f"[{split}] {img_path.name}: missing annotator file(s)"
                )
                total_skipped += 1
                continue

            senior_xy = parse_annotation_file(senior_file)
            junior_xy = parse_annotation_file(junior_file)
            gt_xy = average_annotators(senior_xy, junior_xy)

            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                logging.warning(f"[{split}] {img_path.name}: cv2 failed to read")
                total_skipped += 1
                continue

            orig_h, orig_w = gray.shape[:2]
            coords_norm = normalize_coords(gt_xy, width=orig_w, height=orig_h)
            resized = cv2.resize(
                gray, (target_size, target_size), interpolation=cv2.INTER_LINEAR
            )
            img_chw = to_three_channel(resized)

            pt_data = {
                "image": torch.from_numpy(img_chw),
                "tps": torch.from_numpy(coords_norm.astype(np.float32)),
                "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
                "pixel_spacing": torch.tensor(float(pixel_spacing), dtype=torch.float32),
                "split": split,
            }
            torch.save(pt_data, str(output_dir / split / f"{stem}.pt"))
            total_saved += 1

        logging.info(f"[{split}] done.")

    logging.info(f"All splits done. Saved: {total_saved}, Skipped: {total_skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ISBI 2015 cephalometric images and annotations to .pt files"
    )
    # Layout A (standard ISBI distribution): a single --data-root with
    # TrainingData/Test1Data/Test2Data + AnnotationsByMD/400_senior|400_junior.
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root of the standard ISBI distribution (auto-discovers "
             "TrainingData/Test1Data/Test2Data + AnnotationsByMD).",
    )
    # Layout B (flat): explicit image + annotation dirs, split by filename number.
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--senior-annotations", type=str, default=None)
    parser.add_argument("--junior-annotations", type=str, default=None)
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output root for train/test1/test2 (defaults to --data-root).",
    )
    parser.add_argument("--pixel-spacing", type=float, default=0.1)
    parser.add_argument("--target-size", type=int, default=512)
    args = parser.parse_args()

    if args.data_root:
        # Standard ISBI distribution layout.
        if not Path(args.data_root).exists():
            print(f"ERROR: data root not found: {args.data_root}", file=sys.stderr)
            sys.exit(1)
        preprocess_cephalometric_isbi(
            args.data_root,
            output_dir=args.output_dir,
            pixel_spacing=args.pixel_spacing,
            target_size=args.target_size,
        )
        return

    # Flat layout (requires image-root + both annotation dirs).
    missing = [
        name for name, val in [
            ("--image-root", args.image_root),
            ("--senior-annotations", args.senior_annotations),
            ("--junior-annotations", args.junior_annotations),
            ("--output-dir", args.output_dir),
        ] if not val
    ]
    if missing:
        print(
            f"ERROR: provide --data-root for the standard ISBI layout, or all of "
            f"--image-root/--senior-annotations/--junior-annotations/--output-dir "
            f"for the flat layout. Missing: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not Path(args.image_root).exists():
        print(f"ERROR: image root not found: {args.image_root}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.senior_annotations).exists():
        print(
            f"ERROR: senior annotations dir not found: {args.senior_annotations}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not Path(args.junior_annotations).exists():
        print(
            f"ERROR: junior annotations dir not found: {args.junior_annotations}",
            file=sys.stderr,
        )
        sys.exit(1)

    preprocess_cephalometric(
        args.image_root,
        args.senior_annotations,
        args.junior_annotations,
        args.output_dir,
        pixel_spacing=args.pixel_spacing,
        target_size=args.target_size,
    )


if __name__ == "__main__":
    main()
