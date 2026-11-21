"""CLI: ``python -m card_pipeline run --weights path/to/best.pt`` from id-card-parsing/."""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from .config import PipelineConfig
from .detector import WeightsLoadError
from .evaluation import evaluate_command
from .output import save_predictions
from .recognizer import RecognizerUnavailable
from .runner import run_pipeline

ROOT = Path(__file__).resolve().parent.parent


def run_command(args: argparse.Namespace, *, detector=None, recognizer=None) -> int:
    """Build a complete run, then replace the previous output directory."""
    config = PipelineConfig(
        weights=args.weights, template_path=args.template, data_dir=args.data_dir,
        output_dir=args.output, confidence=args.confidence, canvas_size=args.canvas_size,
        detect_size=args.detect_size, decode_size=args.decode_size,
        template_image=args.template_image, align=not args.no_align,
        device=args.device, ocr_device=args.ocr_device,
        rules_path=args.rules, replace=False)
    output_dir = config.output_dir
    if output_dir == output_dir.parent or output_dir.is_symlink():
        raise ValueError(f"Output path must be a run directory: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and not (output_dir / "predictions.json").is_file():
        raise FileExistsError(f"Output directory is not a previous pipeline run: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-run-",
                                     dir=output_dir.parent) as temporary:
        temporary_dir = Path(temporary)
        staged_dir = temporary_dir / "new"
        staged_config = replace(config, output_dir=staged_dir)
        payload = run_pipeline(staged_config, detector=detector,
                               recognizer=recognizer, scan_ids=args.ids, limit=args.n)
        save_predictions(payload, staged_dir)
        previous_dir = temporary_dir / "previous"
        if output_dir.exists():
            output_dir.rename(previous_dir)
        try:
            staged_dir.rename(output_dir)
        except OSError:
            if previous_dir.exists():
                previous_dir.rename(output_dir)
            raise
    summary = payload["summary"]
    print(f"Processed {summary['scans']} scans: {summary['normalized']} normalized, "
          f"{summary['no_card']} without a card, {summary['degenerate_obb']} degenerate, "
          f"{summary['unreadable_scan']} unreadable, {summary['alignment_failed']} unaligned, "
          f"{summary['other_scan_failures']} other failures.")
    print(f"Fields: {summary['fields_recognized']} recognized, "
          f"{summary['field_failures']} failed.")
    print(f"Checkboxes: {summary['checkboxes_checked']} checked, "
          f"{summary['checkboxes_unchecked']} unchecked, "
          f"{summary['checkboxes_uncertain']} uncertain, "
          f"{summary['fields_deferred']} deferred.")
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
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--ids", nargs="+", help="Scan IDs; numeric IDs accept omitted leading zeros")
    selection.add_argument("--n", type=int, help="Process up to N scans in sorted scan ID order")
    run.add_argument("--confidence", type=float, default=0.25, help="Minimum detection confidence")
    run.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                     help="YOLO device (default: auto)")
    run.add_argument("--ocr-device", choices=("auto", "cpu", "gpu"), default="auto",
                     help="PaddleOCR device (default: auto)")
    run.add_argument("--canvas-size", type=int, help="Override the template's canvas size, "
                     "e.g. 1024 for more pixels per field; boxes are fractions so they still align")
    run.add_argument("--detect-size", type=int, default=1280, help="Detector preview size")
    run.add_argument("--decode-size", type=int, default=4096, help="Scan decode size for cropping")
    run.add_argument("--template-image", type=Path, default=ROOT / "template/template.jpg",
                     help="Blank card image cropped to the card, used to register scans")
    run.add_argument("--no-align", action="store_true",
                     help="Skip form registration (180-degree flips will not be corrected)")
    run.add_argument("--rules", type=Path, help="Optional card_ocr extraction rules JSON")
    run.add_argument("--replace", action="store_true",
                     help="Accepted for compatibility; reruns replace previous output automatically")
    evaluate = sub.add_parser("evaluate", help="Score saved predictions against verified card labels")
    evaluate.add_argument("--predictions", type=Path, required=True, help="Pipeline predictions.json")
    evaluate.add_argument("--labels", type=Path, default=ROOT / "results/verified_labels.json",
                          help="JSON of verified values by scan ID (default: results/verified_labels.json)")
    return parser


def main(argv=None, *, detector=None, recognizer=None) -> int:
    """Parse arguments; report setup errors with a nonzero exit status.

    ``detector`` and ``recognizer`` let tests inject fakes; the CLI normally
    builds the real YOLO-OBB and PaddleOCR adapters.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.command == "evaluate":
            return evaluate_command(args.predictions, args.labels)
        return run_command(args, detector=detector, recognizer=recognizer)
    except (ValueError, OSError, KeyError, TypeError, WeightsLoadError,
            RecognizerUnavailable) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
