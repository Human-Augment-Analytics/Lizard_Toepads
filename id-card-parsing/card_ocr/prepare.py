"""Align cards, crop writing regions, and record original-scan coordinates."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from annotator.geometry import apply_affine, build_transform, invert_affine
from annotator.image_loader import ImageLoader
from annotator.record import load_record
from annotator.template_matcher import CardTemplateMatcher
from .fields import Field
from .preprocessing import FormAligner, clean_printed_form, crop_warnings
from .recipe import preparation_recipe
from .checkboxes import inspect_checkbox


def project(points, matrix) -> np.ndarray:
    """Apply a projective transform to Nx2 pixel coordinates."""
    return cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(1, -1, 2),
                                    np.asarray(matrix, np.float64))[0]


def rectify_card(image: Image.Image, corners: np.ndarray, size: tuple[int, int]):
    """Warp TL/TR/BR/BL corners to an upright card and return its inverse map."""
    corners = np.asarray(corners, np.float32)
    if (corners.shape != (4, 2) or not np.isfinite(corners).all()
            or not cv2.isContourConvex(corners)
            or cv2.contourArea(corners, oriented=True) <= 1):
        raise ValueError("Card corners must be a finite, clockwise convex quadrilateral")
    width, height = size
    if width < 2 or height < 2:
        raise ValueError("Card output must be at least two pixels per side")
    destination = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    matrix = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(np.asarray(image.convert("RGB")), matrix, size,
                                 flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return Image.fromarray(warped), np.linalg.inv(matrix)


def draw_fields(image: Image.Image, fields: list[Field]) -> Image.Image:
    """Render labeled expected writing regions for crop review."""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for field in fields:
        left, top, right, bottom = field.pixel_box(image.size)
        draw.rectangle((left, top, right-1, bottom-1), outline="#0078ff", width=2)
        draw.text((left, max(0, top-13)), field.name, fill="#0044aa")
    return overlay


def _original_corners(image_id: str, path: Path, annotations: Path, template: Path,
                      matcher: list) -> tuple[np.ndarray, str]:
    """Get scan-space corners from a saved annotation or cached template matcher."""
    record = load_record(image_id, annotations)
    if record is not None:
        if record.image_id != image_id:
            raise ValueError("Annotation image ID does not match the scan")
        matrix = np.asarray(record.transform_inverse, float)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            raise ValueError("Invalid annotation inverse transform")
        return apply_affine(matrix, record.obb_corners), "annotation"
    if not matcher:
        matcher.append(CardTemplateMatcher(template))
    preview = ImageLoader(1024).load(path)
    match = matcher[0].match(preview.preview)
    if match is None:
        raise ValueError("No reliable card detection; annotate this image first")
    inverse = invert_affine(build_transform(preview.scale, preview.offset))
    return apply_affine(inverse, match.obb.corners()), "template"


def prepare_run(data_dir: Path, annotations_dir: Path, template: Path,
                fields: list[Field], image_ids: list[str], output: Path,
                *, card_width: int = 1000, decode_size: int = 4096,
                clean_print: bool = True, checkboxes: list[Field] | None = None,
                replace: bool = False) -> dict:
    """Generate a complete run before replacing existing crops and predictions."""
    output = Path(output)
    if output.exists() and not replace:
        raise FileExistsError(f"Output already exists: {output}; use --replace to rebuild")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / "run"
        result = _prepare_run(data_dir, annotations_dir, template, fields, image_ids, staging,
                              card_width=card_width, decode_size=decode_size,
                              clean_print=clean_print, checkboxes=checkboxes)
        if result["errors"]:
            failures = "; ".join(f"{row['image_id']}: {row['error']}" for row in result["errors"])
            raise ValueError(f"Crop preparation failed; existing results preserved: {failures}")
        backup = Path(temporary) / "previous"
        if output.exists():
            output.rename(backup)
        try:
            staging.rename(output)
        except OSError:
            if backup.exists():
                backup.rename(output)
            raise
    return result


def _prepare_run(data_dir: Path, annotations_dir: Path, template: Path,
                fields: list[Field], image_ids: list[str], output: Path,
                *, card_width: int = 1000, decode_size: int = 4096,
                clean_print: bool = True,
                checkboxes: list[Field] | None = None) -> dict:
    """Prepare crops and coordinate maps in a new run directory.

    Record per-image failures alongside successfully prepared crops.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new run directory")
    if not 100 <= card_width <= 3000 or not card_width <= decode_size <= 8192:
        raise ValueError("Use card width 100–3000 and decode size between card width and 8192")
    with Image.open(template) as image:
        card_size = (card_width, round(card_width * image.height / image.width))
        template_image = image.convert("RGB").resize(card_size, Image.Resampling.LANCZOS)
    aligner = FormAligner(template_image)
    checkboxes = checkboxes or []
    output.mkdir(parents=True)
    draw_fields(template_image, fields).save(output / "template_fields.png")
    paths = {}
    for path in sorted(Path(data_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            if path.stem in paths:
                raise ValueError(f"Multiple images have ID {path.stem}")
            paths[path.stem] = path
    manifest = {"version": 2, "card_size": list(card_size), "samples": [], "errors": [],
                "recipe": preparation_recipe(template, fields, card_width=card_width,
                                             decode_size=decode_size, clean_print=clean_print,
                                             checkboxes=checkboxes),
                "preprocessing": {"alignment": "printed_form_sift_ransac", "clean_print": clean_print,
                                  "opencv_version": cv2.__version__},
                "images": [], "checkboxes": []}
    matcher = []
    for image_id in image_ids:
        print(f"Preparing {image_id}…", flush=True)
        try:
            if image_id not in paths:
                raise FileNotFoundError(f"Image ID not found: {image_id}")
            path = paths[image_id]
            corners, source = _original_corners(image_id, path, annotations_dir, template, matcher)
            with Image.open(path) as image:
                original_width, original_height = image.size
                if (not np.isfinite(corners).all() or np.any(corners < -2)
                        or np.any(corners > np.array(image.size) + 2)):
                    raise ValueError("Annotation extends outside this scan; review its box")
                # JPEG draft reduces decoding cost before allocating RGB pixels.
                image.draft("RGB", (decode_size, decode_size))
                image.thumbnail((decode_size, decode_size), Image.Resampling.LANCZOS)
                decoded = image.convert("RGB")
            scale = np.diag([decoded.width/original_width, decoded.height/original_height, 1])
            decoded_corners = corners * np.diag(scale)[:2]
            coarse_card, card_to_decoded = rectify_card(decoded, decoded_corners, card_size)
            registration = aligner.register(coarse_card)
            card_to_decoded = card_to_decoded @ np.linalg.inv(registration.card_to_template)
            card_to_original = np.linalg.inv(scale) @ card_to_decoded
            # Warp from the decoded scan once, avoiding a second resampling of handwriting.
            card = Image.fromarray(cv2.warpPerspective(np.array(decoded), np.linalg.inv(card_to_decoded),
                                                       card_size, flags=cv2.INTER_CUBIC,
                                                       borderValue=(255, 255, 255)))
            header_box = next((field.pixel_box(card_size) for field in fields
                               if field.name == 'header_notes'), None)
            cleaned, removed = clean_printed_form(card, template_image,
                                                  header_notes_box=header_box) if clean_print else (
                card.copy(), Image.new('L', card_size, 0))
            image_output = output / image_id
            image_output.mkdir()
            coarse_card.save(image_output / "card_unaligned.png")
            card.save(image_output / "card.png")
            cleaned.save(image_output / "card_clean.png")
            removed.save(image_output / "print_removed.png")
            draw_fields(card, fields).save(image_output / "fields.png")
            draw_fields(card, checkboxes).save(image_output / 'checkboxes.png')
            for box in checkboxes:
                region = box.pixel_box(card_size)
                checkbox_crop = card.crop(region)
                checkbox_path = image_output / f'checkbox_{box.name}.png'
                checkbox_crop.save(checkbox_path)
                result = inspect_checkbox(checkbox_crop, template_image.crop(region))
                manifest['checkboxes'].append(dict(
                    image_id=image_id, field=box.name, crop=checkbox_path.relative_to(output).as_posix(),
                    card_box=list(region), **result))
            manifest["images"].append(dict(image_id=image_id, alignment=registration.diagnostics))
            overview = decoded.copy()
            draw = ImageDraw.Draw(overview)
            for field in fields:
                left, top, right, bottom = field.pixel_box(card_size)
                polygon = [[left, top], [right-1, top], [right-1, bottom-1], [left, bottom-1]]
                original_polygon = project(polygon, card_to_original)
                display_polygon = project(polygon, card_to_decoded)
                draw.polygon([tuple(point) for point in display_polygon], outline="#0078ff", width=3)
                crop_path = image_output / f"{field.name}.png"
                raw_path = image_output / f"{field.name}_raw.png"
                region = cleaned.crop((left, top, right, bottom))
                warnings = crop_warnings(region)
                if field.review_note:
                    warnings.append(field.review_note)
                ImageOps.expand(region, border=8, fill="white").save(crop_path)
                ImageOps.expand(card.crop((left, top, right, bottom)), border=8, fill="white").save(raw_path)
                manifest["samples"].append({
                    "image_id": image_id, "field": field.name,
                    "crop": crop_path.relative_to(output).as_posix(), "source": source,
                    "raw_crop": raw_path.relative_to(output).as_posix(), "crop_warnings": warnings,
                    "removed_print_pixels": int(np.count_nonzero(np.array(removed.crop((left, top, right, bottom))))),
                    "card_box": [left, top, right, bottom], "crop_padding": 8,
                    "original_size": [original_width, original_height],
                    "original_polygon": original_polygon.tolist(),
                    "card_to_original": card_to_original.tolist(),
                })
            overview.thumbnail((1200, 1200))
            overview.save(image_output / "scan_fields.jpg")
        except (ValueError, OSError, KeyError, TypeError, cv2.error) as exc:
            manifest["errors"].append({"image_id": image_id, "error": str(exc)})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
