"""CLI: ``python -m card_pipeline run --weights path/to/best.pt`` from id-card-parsing/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import PipelineConfig
from .detector import WeightsLoadError
from .output import save_predictions
from .recognizer import RecognizerUnavailable
from .runner import run_pipeline

ROOT = Path(__file__).resolve().parent.parent


def run_command(args: argparse.Namespace, *, detector=None, recognizer=None) -> int:
    """Run the pipeline, save predictions, and return 1 if anything failed."""
    config = PipelineConfig(
        weights=args.weights, template_path=args.template, data_dir=args.data_dir,
        output_dir=args.output, confidence=args.confidence, canvas_size=args.canvas_size,
        detect_size=args.detect_size, decode_size=args.decode_size,
        template_image=args.template_image, align=not args.no_align,
        rules_path=args.rules, replace=args.replace)
    payload = run_pipeline(config, detector=detector, recognizer=recognizer, scan_ids=args.ids)
    save_predictions(payload, config.output_dir, replace=config.replace)
    summary = payload["summary"]
    print(f"Processed {summary['scans']} scans: {summary['normalized']} normalized, "
          f"{summary['no_card']} without a card, {summary['degenerate_obb']} degenerate, "
          f"{summary['unreadable_scan']} unreadable, {summary['alignment_failed']} unaligned, "
          f"{summary['other_scan_failures']} other failures.")
    print(f"Fields: {summary['fields_recognized']} recognized, "
          f"{summary['fields_deferred']} checkbox regions deferred, "
          f"{summary['field_failures']} failed.")
    print(f"Saved predictions: {config.output_dir}")
    failed = (summary["normalized"] != summary["scans"] or summary["field_failures"])
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Detect cards, crop template fields, and read them with PaddleOCR")
    run.add_argument("--weights", type=Path, required=True, help="YOLO-OBB weights (.pt)")
    run.add_argument("--template", type=Path, default=ROOT / "template/field_template.json")
    run.add_argument("--data-dir", type=Path, default=ROOT / "data")
    run.add_argument("--output", type=Path, default=ROOT / "results/pipeline/current")
    run.add_argument("--ids", nargs="+", help="Scan IDs; numeric IDs accept omitted leading zeros")
    run.add_argument("--confidence", type=float, default=0.25, help="Minimum detection confidence")
    run.add_argument("--canvas-size", type=int, help="Override the template's canvas size, "
                     "e.g. 1024 for more pixels per field; boxes are fractions so they still align")
    run.add_argument("--detect-size", type=int, default=1280, help="Detector preview size")
    run.add_argument("--decode-size", type=int, default=4096, help="Scan decode size for cropping")
    run.add_argument("--template-image", type=Path, default=ROOT / "template/template.jpg",
                     help="Blank card image cropped to the card, used to register scans")
    run.add_argument("--no-align", action="store_true",
                     help="Skip form registration (180-degree flips will not be corrected)")
    run.add_argument("--rules", type=Path, help="Optional card_ocr extraction rules JSON")
    run.add_argument("--replace", action="store_true", help="Overwrite existing predictions")
    return parser


def main(argv=None, *, detector=None, recognizer=None) -> int:
    """Parse arguments; report setup errors with a nonzero exit status.

    ``detector`` and ``recognizer`` let tests inject fakes; the CLI normally
    builds the real YOLO-OBB and PaddleOCR adapters.
    """
    args = build_parser().parse_args(argv)
    try:
        return run_command(args, detector=detector, recognizer=recognizer)
    except (ValueError, OSError, KeyError, TypeError, WeightsLoadError,
            RecognizerUnavailable) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
