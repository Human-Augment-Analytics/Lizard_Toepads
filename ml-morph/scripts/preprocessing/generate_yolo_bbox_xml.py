#!/usr/bin/env python3
"""
Replace tight landmark-derived bounding boxes in XML with YOLO-derived bounding boxes.

The dlib shape predictor is sensitive to bounding box size. At inference, it receives
YOLO detection boxes (much larger than landmark+20px). This script generates training
XMLs with YOLO-like bounding boxes so training matches inference conditions.

When using OBB models with --crops-dir, images are cropped around the OBB center and
rotated so the OBB becomes upright, producing much tighter bounding boxes. GT landmarks
are transformed to match the rotated crop coordinates.

GT landmarks are preserved (transformed if rotating); only the bounding boxes are replaced.
"""

import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

import cv2
import numpy as np


def find_best_toe_detection(results, gt_centroid, target_class="toe"):
    """Find the YOLO detection whose center is closest to GT landmark centroid.

    Supports both regular detection (res.boxes) and OBB detection (res.obb).
    Returns (axis_aligned_box, xywhr_or_None).
      - axis_aligned_box: [x1, y1, x2, y2] axis-aligned bounding rect
      - xywhr: [cx, cy, w, h, angle_rad] from OBB, or None for regular boxes
    """
    best_box = None
    best_xywhr = None
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
                # xywhr: [cx, cy, w, h, angle_rad]
                xywhr = res.obb.xywhr[i].tolist()
                best_xywhr = xywhr
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
                best_xywhr = None

    return best_box, best_xywhr


def crop_and_rotate(img, cx, cy, obb_w, obb_h, angle_rad, crop_scale=2.0):
    """Crop region around OBB center and rotate so OBB becomes upright.

    Args:
        img: Full image (numpy array, HxWxC)
        cx, cy: OBB center in image coordinates
        obb_w, obb_h: OBB width and height
        angle_rad: OBB rotation angle in radians
        crop_scale: How much larger than OBB to make the initial crop (default 2.0)

    Returns:
        rotated_crop: The rotated crop image
        rot_matrix: 2x3 affine rotation matrix (for transforming points)
        crop_offset: (ox, oy) offset of crop region in original image
        crop_size: (crop_w, crop_h) size of the crop before rotation
    """
    img_h, img_w = img.shape[:2]

    # Crop region: crop_scale * max(obb_w, obb_h) centered on OBB center
    crop_half = int(crop_scale * max(obb_w, obb_h) / 2)

    # Crop bounds, clamped to image
    crop_x1 = max(0, int(cx - crop_half))
    crop_y1 = max(0, int(cy - crop_half))
    crop_x2 = min(img_w, int(cx + crop_half))
    crop_y2 = min(img_h, int(cy + crop_half))

    crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_h_actual, crop_w_actual = crop.shape[:2]

    # OBB center relative to crop
    cx_crop = cx - crop_x1
    cy_crop = cy - crop_y1

    # Rotation angle: negate so OBB becomes upright
    angle_deg = -np.degrees(angle_rad)

    # Rotation matrix around OBB center in crop coordinates
    rot_matrix = cv2.getRotationMatrix2D((cx_crop, cy_crop), angle_deg, 1.0)

    # Compute new bounding size after rotation to avoid clipping
    cos_a = abs(np.cos(np.radians(angle_deg)))
    sin_a = abs(np.sin(np.radians(angle_deg)))
    new_w = int(crop_w_actual * cos_a + crop_h_actual * sin_a)
    new_h = int(crop_h_actual * cos_a + crop_w_actual * sin_a)

    # Adjust rotation matrix for the new canvas center
    rot_matrix[0, 2] += (new_w - crop_w_actual) / 2
    rot_matrix[1, 2] += (new_h - crop_h_actual) / 2

    rotated_crop = cv2.warpAffine(crop, rot_matrix, (new_w, new_h))

    crop_offset = (crop_x1, crop_y1)
    crop_size = (crop_w_actual, crop_h_actual)

    return rotated_crop, rot_matrix, crop_offset, crop_size


def transform_landmarks(landmarks, rot_matrix, crop_offset):
    """Transform landmark coordinates from original image to rotated crop.

    Args:
        landmarks: Nx2 array of (x, y) in original image coords
        rot_matrix: 2x3 affine matrix from crop_and_rotate
        crop_offset: (ox, oy) crop origin in original image

    Returns:
        Nx2 array of transformed landmark coordinates
    """
    ox, oy = crop_offset
    # Shift landmarks to crop-local coords
    pts = landmarks.copy().astype(np.float64)
    pts[:, 0] -= ox
    pts[:, 1] -= oy

    # Apply rotation: [x', y'] = M @ [x, y, 1]
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])  # Nx3
    transformed = (rot_matrix @ pts_h.T).T  # Nx2

    return transformed


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
    padding_ratio=0.1,
    fallback_padding_ratio=0.5,
    conf_threshold=0.25,
    target_class="toe",
    crops_dir=None,
    crop_scale=2.0,
    no_rotation=False,
):
    from ultralytics import YOLO

    tree = ET.parse(input_xml)
    root = tree.getroot()
    images_elem = root.find("images")

    model = YOLO(yolo_model_path)

    # Create crops directory if doing rotation
    if crops_dir and not no_rotation:
        crops_path = Path(crops_dir)
        crops_path.mkdir(parents=True, exist_ok=True)

    yolo_count = 0
    rotated_count = 0
    fallback_count = 0
    total = 0

    for image_elem in images_elem.findall("image"):
        img_path = image_elem.get("file")

        for box_elem in image_elem.findall("box"):
            total += 1

            # Extract GT landmarks (unchanged for non-rotation path)
            landmarks = []
            for part in box_elem.findall("part"):
                landmarks.append((int(part.get("x")), int(part.get("y"))))
            landmarks = np.array(landmarks, dtype=np.float64)
            gt_centroid = landmarks.mean(axis=0)

            # Load image with cv2
            img = cv2.imread(img_path)
            if img is None:
                print(f"  WARNING: Could not read {img_path}, using fallback")
                img_h, img_w = 10000, 20000  # rough fallback dims
            else:
                img_h, img_w = img.shape[:2]

            # Run YOLO
            results = model(img_path, conf=conf_threshold, device=0, verbose=False)
            best_box, best_xywhr = find_best_toe_detection(
                results, gt_centroid, target_class
            )

            if best_box is not None and best_xywhr is not None and not no_rotation and crops_dir and img is not None:
                # OBB crop+rotate path
                cx, cy, obb_w, obb_h, angle_rad = best_xywhr

                rotated_crop, rot_matrix, crop_offset, crop_size = crop_and_rotate(
                    img, cx, cy, obb_w, obb_h, angle_rad, crop_scale
                )

                # Transform GT landmarks to rotated crop coords
                transformed_lm = transform_landmarks(landmarks, rot_matrix, crop_offset)

                # Save rotated crop
                stem = Path(img_path).stem
                # Use box index to handle multiple detections per image
                crop_filename = f"{stem}_det{total}.jpg"
                crop_path = str(Path(crops_dir) / crop_filename)
                cv2.imwrite(crop_path, rotated_crop)

                # Compute tight bbox from OBB dimensions + padding in rotated crop
                rot_h, rot_w = rotated_crop.shape[:2]
                # OBB center in rotated crop: apply same transform
                obb_center = np.array([[cx, cy]], dtype=np.float64)
                obb_center_rot = transform_landmarks(obb_center, rot_matrix, crop_offset)[0]

                # Tight box from OBB w,h centered at rotated OBB center
                half_w = obb_w / 2
                half_h = obb_h / 2
                tight_x1 = obb_center_rot[0] - half_w
                tight_y1 = obb_center_rot[1] - half_h
                tight_x2 = obb_center_rot[0] + half_w
                tight_y2 = obb_center_rot[1] + half_h

                # Add padding
                new_left, new_top, new_right, new_bottom = expand_box(
                    tight_x1, tight_y1, tight_x2, tight_y2, padding_ratio, rot_w, rot_h
                )

                new_w = new_right - new_left
                new_h = new_bottom - new_top

                # Update XML: change image path to crop, update box, update landmarks
                image_elem.set("file", crop_path)
                box_elem.set("left", str(new_left))
                box_elem.set("top", str(new_top))
                box_elem.set("width", str(new_w))
                box_elem.set("height", str(new_h))

                # Update landmark coordinates
                for j, part in enumerate(box_elem.findall("part")):
                    part.set("x", str(int(round(transformed_lm[j, 0]))))
                    part.set("y", str(int(round(transformed_lm[j, 1]))))

                rotated_count += 1
                source = "OBB-rotated"

                print(
                    f"  {Path(img_path).stem}: {source} bbox "
                    f"left={new_left} top={new_top} w={new_w} h={new_h} "
                    f"angle={np.degrees(angle_rad):.1f}deg crop={crop_filename}"
                )

            elif best_box is not None:
                # Axis-aligned fallback (regular detection or --no-rotation)
                x1, y1, x2, y2 = best_box
                new_left, new_top, new_right, new_bottom = expand_box(
                    x1, y1, x2, y2, padding_ratio, img_w, img_h
                )
                yolo_count += 1
                source = "YOLO-aligned"

                new_w = new_right - new_left
                new_h = new_bottom - new_top

                box_elem.set("left", str(new_left))
                box_elem.set("top", str(new_top))
                box_elem.set("width", str(new_w))
                box_elem.set("height", str(new_h))

                print(
                    f"  {Path(img_path).stem}: {source} bbox "
                    f"left={new_left} top={new_top} w={new_w} h={new_h}"
                )

            else:
                new_left, new_top, new_right, new_bottom = fallback_box(
                    landmarks, fallback_padding_ratio, img_w, img_h
                )
                fallback_count += 1
                source = "fallback"

                new_w = new_right - new_left
                new_h = new_bottom - new_top

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
    print(f"  OBB-rotated:  {rotated_count} ({100*rotated_count/max(total,1):.0f}%)")
    print(f"  YOLO-aligned: {yolo_count} ({100*yolo_count/max(total,1):.0f}%)")
    print(f"  Fallback:     {fallback_count} ({100*fallback_count/max(total,1):.0f}%)")
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
        default=0.1,
        help="Padding ratio for YOLO boxes (default 0.1)",
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
    parser.add_argument(
        "--crops-dir",
        type=str,
        default=None,
        help="Directory to save rotated OBB crops (enables crop+rotate mode)",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=2.0,
        help="Crop region size as multiple of OBB size (default 2.0)",
    )
    parser.add_argument(
        "--no-rotation",
        action="store_true",
        help="Disable crop+rotate, use axis-aligned boxes only (old behavior)",
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
        crops_dir=args.crops_dir,
        crop_scale=args.crop_scale,
        no_rotation=args.no_rotation,
    )


if __name__ == "__main__":
    main()
