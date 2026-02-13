#!/usr/bin/env python3
"""
Replace tight landmark-derived bounding boxes in XML with YOLO-derived bounding boxes.

The dlib shape predictor is sensitive to bounding box size. At inference, it receives
YOLO detection boxes (much larger than landmark+20px). This script generates training
XMLs with YOLO-like bounding boxes so training matches inference conditions.

GT landmarks are preserved unchanged; only the bounding boxes are replaced.
"""

import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

import numpy as np
from PIL import Image

# Allow very large images (normalized images are ~10k x 10k)
Image.MAX_IMAGE_PIXELS = None


def find_best_toe_detection(results, gt_centroid, target_class="toe"):
    """Find the YOLO detection whose center is closest to GT landmark centroid.

    Supports both regular detection (res.boxes) and OBB detection (res.obb).
    For OBB, the rotated box is converted to an axis-aligned bounding rectangle.
    """
    best_box = None
    best_dist = float("inf")

    res = results[0]

    # Determine if results are OBB or regular
    if res.obb is not None and len(res.obb) > 0:
        # OBB model: use res.obb
        for i in range(len(res.obb)):
            cls_name = res.names[int(res.obb.cls[i].item())]
            if target_class not in cls_name.lower():
                continue

            # xyxyxyxy gives 4 corner points as [N, 4, 2]
            corners = res.obb.xyxyxyxy[i].tolist()  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            # Convert to axis-aligned bbox
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            dist = np.sqrt((cx - gt_centroid[0]) ** 2 + (cy - gt_centroid[1]) ** 2)

            if dist < best_dist:
                best_dist = dist
                best_box = [x1, y1, x2, y2]
    elif res.boxes is not None and len(res.boxes) > 0:
        # Regular detection: use res.boxes
        for box in res.boxes:
            cls_name = res.names[int(box.cls[0].item())]
            if target_class not in cls_name.lower():
                continue

            xyxy = box.xyxy[0].tolist()
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            dist = np.sqrt((cx - gt_centroid[0]) ** 2 + (cy - gt_centroid[1]) ** 2)

            if dist < best_dist:
                best_dist = dist
                best_box = xyxy

    return best_box


def expand_box(x1, y1, x2, y2, padding_ratio, img_w, img_h):
    """Expand box by padding_ratio on each side, clamped to image bounds."""
    w_box = x2 - x1
    h_box = y2 - y1
    px = int(w_box * padding_ratio)
    py = int(h_box * padding_ratio)
    return (
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(img_w, int(x2 + px)),
        min(img_h, int(y2 + py)),
    )


def fallback_box(landmarks, fallback_padding_ratio, img_w, img_h):
    """Create a proportional-padding box from landmarks when YOLO fails."""
    lm_min_x, lm_min_y = landmarks.min(axis=0)
    lm_max_x, lm_max_y = landmarks.max(axis=0)
    lm_w = lm_max_x - lm_min_x
    lm_h = lm_max_y - lm_min_y
    pad_x = int(lm_w * fallback_padding_ratio)
    pad_y = int(lm_h * fallback_padding_ratio)
    return (
        max(0, int(lm_min_x - pad_x)),
        max(0, int(lm_min_y - pad_y)),
        min(img_w, int(lm_max_x + pad_x)),
        min(img_h, int(lm_max_y + pad_y)),
    )


def generate_yolo_bbox_xml(
    input_xml,
    output_xml,
    yolo_model_path,
    padding_ratio=0.3,
    fallback_padding_ratio=0.5,
    conf_threshold=0.25,
    target_class="toe",
):
    from ultralytics import YOLO

    tree = ET.parse(input_xml)
    root = tree.getroot()
    images_elem = root.find("images")

    model = YOLO(yolo_model_path)

    yolo_count = 0
    fallback_count = 0
    total = 0

    for image_elem in images_elem.findall("image"):
        img_path = image_elem.get("file")

        for box_elem in image_elem.findall("box"):
            total += 1

            # Extract GT landmarks (unchanged)
            landmarks = []
            for part in box_elem.findall("part"):
                landmarks.append((int(part.get("x")), int(part.get("y"))))
            landmarks = np.array(landmarks, dtype=np.float64)
            gt_centroid = landmarks.mean(axis=0)

            # Get image dimensions
            img_pil = Image.open(img_path)
            img_w, img_h = img_pil.size

            # Run YOLO
            results = model(img_pil, conf=conf_threshold, device=0, verbose=False)
            best_box = find_best_toe_detection(results, gt_centroid, target_class)

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                new_left, new_top, new_right, new_bottom = expand_box(
                    x1, y1, x2, y2, padding_ratio, img_w, img_h
                )
                yolo_count += 1
                source = "YOLO"
            else:
                new_left, new_top, new_right, new_bottom = fallback_box(
                    landmarks, fallback_padding_ratio, img_w, img_h
                )
                fallback_count += 1
                source = "fallback"

            new_w = new_right - new_left
            new_h = new_bottom - new_top

            # Update box element
            box_elem.set("left", str(new_left))
            box_elem.set("top", str(new_top))
            box_elem.set("width", str(new_w))
            box_elem.set("height", str(new_h))

            print(
                f"  {Path(img_path).stem}: {source} bbox "
                f"left={new_left} top={new_top} w={new_w} h={new_h}"
            )

    # Pretty print and write
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(output_xml, "w") as f:
        f.write(xml_str)

    print(f"\nSummary: {total} images processed")
    print(f"  YOLO detections: {yolo_count} ({100*yolo_count/total:.0f}%)")
    print(f"  Fallback:        {fallback_count} ({100*fallback_count/total:.0f}%)")
    print(f"  Output: {output_xml}")


def main():
    parser = argparse.ArgumentParser(
        description="Replace XML bounding boxes with YOLO-derived boxes"
    )
    parser.add_argument("--input-xml", required=True, help="Input XML (e.g. train.xml)")
    parser.add_argument("--output-xml", required=True, help="Output XML path")
    parser.add_argument("--yolo-model", required=True, help="Path to YOLO .pt model")
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.3,
        help="Padding ratio for YOLO boxes (default 0.3, matching inference)",
    )
    parser.add_argument(
        "--fallback-padding-ratio",
        type=float,
        default=0.5,
        help="Proportional padding when YOLO misses (default 0.5)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default 0.25)",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default="toe",
        help="YOLO class to match (default: toe)",
    )
    args = parser.parse_args()

    generate_yolo_bbox_xml(
        args.input_xml,
        args.output_xml,
        args.yolo_model,
        padding_ratio=args.padding_ratio,
        fallback_padding_ratio=args.fallback_padding_ratio,
        conf_threshold=args.conf_threshold,
        target_class=args.target_class,
    )


if __name__ == "__main__":
    main()
