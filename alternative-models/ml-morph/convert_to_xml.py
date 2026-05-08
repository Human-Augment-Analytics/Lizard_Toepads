import sys
import os
import json
import argparse
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

import cv2
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

ML_MORPH_DIR = Path(__file__).parent.resolve()
SHARED_DATA_DIR = "/storage/ice-shared/cs8903onl/alternative-models/data"


def generate_xml(entries, output_path, dataset_name="ml-morph"):
    root = ET.Element("dataset")
    name_el = ET.SubElement(root, "name")
    name_el.text = dataset_name
    ET.SubElement(root, "comment")
    images_el = ET.SubElement(root, "images")

    for jpg_path, landmarks in entries:
        image_el = ET.SubElement(images_el, "image")
        image_el.set("file", str(jpg_path))

        box_el = ET.SubElement(image_el, "box")
        box_el.set("top", "0")
        box_el.set("left", "0")
        box_el.set("width", "512")
        box_el.set("height", "512")

        for i, (x, y) in enumerate(landmarks):
            part_el = ET.SubElement(box_el, "part")
            part_el.set("name", str(i))
            part_el.set("x", str(int(round(x))))
            part_el.set("y", str(int(round(y))))

    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(output_path, "w") as f:
        f.write(xmlstr)


def process_pt_files(pt_paths, images_dir):
    entries = []
    skipped = 0
    for pt_path in pt_paths:
        pt_path = Path(pt_path)
        try:
            data = torch.load(str(pt_path), map_location="cpu")
        except Exception as e:
            logging.warning(f"Failed to load {pt_path.name}: {e}")
            skipped += 1
            continue

        if "image" not in data or "tps" not in data:
            logging.warning(f"Skipping {pt_path.name}: missing 'image' or 'tps' key")
            skipped += 1
            continue

        img_tensor = data["image"]
        tps_tensor = data["tps"]

        if img_tensor.shape != (3, 512, 512):
            logging.warning(f"Skipping {pt_path.name}: unexpected image shape {img_tensor.shape}")
            skipped += 1
            continue

        if tps_tensor.shape != (9, 2):
            logging.warning(f"Skipping {pt_path.name}: unexpected tps shape {tps_tensor.shape}")
            skipped += 1
            continue

        img_hwc = img_tensor.permute(1, 2, 0).numpy()
        stem = pt_path.stem
        jpg_path = images_dir / f"{stem}.jpg"
        cv2.imwrite(str(jpg_path), img_hwc)

        landmarks = tps_tensor.numpy().tolist()
        entries.append((str(jpg_path), landmarks))

    return entries, skipped


def main():
    parser = argparse.ArgumentParser(description="Convert shared .pt crop files to dlib XML format")
    parser.add_argument("--split", required=True, help="Path to split.json")
    parser.add_argument("--data-dir", type=str, default=SHARED_DATA_DIR, help="Shared data root directory")
    args = parser.parse_args()

    split_path = Path(args.split)
    if not split_path.exists():
        print(f"ERROR: split file not found: {split_path}", file=sys.stderr)
        sys.exit(1)

    with open(split_path) as f:
        split_data = json.load(f)

    if "train" not in split_data or "val" not in split_data:
        print("ERROR: split.json missing 'train' or 'val' keys", file=sys.stderr)
        sys.exit(1)

    data_dir = ML_MORPH_DIR / "data"
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Processing train split ({len(split_data['train'])} files)...")
    train_entries, train_skipped = process_pt_files(split_data["train"], images_dir)
    generate_xml(train_entries, data_dir / "train.xml", "ml-morph-train")
    logging.info(f"  train.xml: {len(train_entries)} images, {train_skipped} skipped")

    logging.info(f"Processing val split ({len(split_data['val'])} files)...")
    val_entries, val_skipped = process_pt_files(split_data["val"], images_dir)
    generate_xml(val_entries, data_dir / "val.xml", "ml-morph-val")
    logging.info(f"  val.xml: {len(val_entries)} images, {val_skipped} skipped")

    test_dir = Path(args.data_dir) / "test"
    if test_dir.exists():
        test_files = sorted(test_dir.glob("*.pt"))
        logging.info(f"Processing test set ({len(test_files)} files)...")
        test_entries, test_skipped = process_pt_files([str(f) for f in test_files], images_dir)
        generate_xml(test_entries, data_dir / "test.xml", "ml-morph-test")
        logging.info(f"  test.xml: {len(test_entries)} images, {test_skipped} skipped")
    else:
        logging.warning(f"Test directory not found: {test_dir}, skipping test.xml")

    total = len(train_entries) + len(val_entries)
    logging.info(f"Conversion complete. {total} images saved to {images_dir}")


if __name__ == "__main__":
    main()
