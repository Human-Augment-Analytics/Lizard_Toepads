"""Prepare handwriting crops and run PaddleOCR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .images import select_images
from .extraction import DEFAULT_RULES

ROOT = Path(__file__).resolve().parent.parent


def prepare_command(args: argparse.Namespace) -> int:
    """Prepare selected fields and return a failure status if any image fails."""
    from .fields import load_layout, load_checkboxes
    from .prepare import prepare_run
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    fields = load_layout(args.layout)
    if args.fields:
        missing = set(args.fields) - {field.name for field in fields}
        if missing:
            raise ValueError(f"Unknown fields: {sorted(missing)}")
        fields = [field for field in fields if field.name in args.fields]
    ids = select_images(args.data_dir, ids=args.ids, count=args.limit, require_count=False)
    result = prepare_run(args.data_dir, args.annotations_dir, args.template,
                         fields, ids, args.output, card_width=args.card_width,
                         decode_size=args.decode_size, clean_print=not args.keep_print,
                         checkboxes=load_checkboxes(args.layout), replace=args.replace)
    print(f"Prepared {len(result['samples'])} fields; {len(result['errors'])} images failed. Output: {args.output}")
    for error in result["errors"]:
        print(f"  {error['image_id']}: {error['error']}")
    return 1 if result["errors"] else 0


def run_command(args: argparse.Namespace) -> int:
    """Save CPU PaddleOCR predictions and report recognition failures."""
    from .models import DEFAULT_CACHE
    from .runner import run_model, save_predictions
    output = args.output or args.manifest.parent / "predictions/paddle/results.json"
    if not args.replace and (output.exists() or output.with_suffix(".csv").exists()):
        raise FileExistsError(f"Output already exists: {output}")
    result = run_model(args.manifest, fields=args.fields, rules_path=args.rules,
                       cache=args.cache_dir or DEFAULT_CACHE, device="cpu",
                       model_id=args.model_id)
    failures = sum(row["status"] != "ok" for row in result["predictions"])
    if args.replace and failures:
        print("Recognition failed; existing predictions were preserved.", file=sys.stderr)
        print(next(row["error"] for row in result["predictions"] if row["status"] != "ok"), file=sys.stderr)
        return 1
    save_predictions(result, output, replace=args.replace)
    print(f"Saved predictions: {output}; {failures} fields failed")
    if failures:
        print(next(row["error"] for row in result["predictions"] if row["status"] != "ok"))
    return 1 if failures else 0


def main(argv=None) -> int:
    """Parse CLI arguments and report command errors with a nonzero exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Export field crops and coordinate overlays without OCR models")
    prepare.add_argument("--ids", nargs="+", help="Image IDs; numeric IDs accept omitted leading zeros")
    prepare.add_argument("--limit", type=int, default=3, help="Default image count when --ids is absent")
    prepare.add_argument("--data-dir", type=Path, default=ROOT / "data")
    prepare.add_argument("--annotations-dir", type=Path, default=ROOT / "annotations")
    prepare.add_argument("--template", type=Path, default=ROOT / "template/template.jpg")
    prepare.add_argument("--layout", type=Path, default=ROOT / "template/fields.json")
    prepare.add_argument("--fields", nargs="+")
    prepare.add_argument("--card-width", type=int, default=1000)
    prepare.add_argument("--decode-size", type=int, default=4096)
    prepare.add_argument("--keep-print", action="store_true", help="Keep printing in OCR inputs; still align the form and save raw crops")
    prepare.add_argument("--output", type=Path, default=ROOT / "results/ocr/current")
    prepare.add_argument("--replace", action="store_true", help="Rebuild crops before replacing existing output")
    recognize = sub.add_parser("run", help="Recognize the prepared crops with PaddleOCR")
    recognize.add_argument("--manifest", type=Path, required=True)
    recognize.add_argument("--model-id", help="Optional PaddleOCR checkpoint override")
    recognize.add_argument("--cache-dir", type=Path)
    recognize.add_argument("--fields", nargs="+")
    recognize.add_argument("--rules", type=Path, default=DEFAULT_RULES, help="Field extraction patterns and output formats")
    recognize.add_argument("--output", type=Path)
    recognize.add_argument("--replace", action="store_true", help="Replace predictions only after successful recognition")
    args = parser.parse_args(argv)
    handlers = {"prepare": prepare_command, "run": run_command}
    try:
        return handlers[args.command](args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
