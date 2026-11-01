"""Find local OCR runs and execute PaddleOCR in an interruptible child process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess

from .extraction import DEFAULT_RULES


def choose_run(project: Path, image_ids, fields, *, start_new=False, review=False, recipe=None) -> Path:
    """Select the managed current run without creating or deleting any files."""
    run = Path(project) / "results/ocr/current"
    expected = {(image_id, field) for image_id in image_ids for field in fields}
    manifest_path = run / "manifest.json"
    if manifest_path.is_file() and not start_new:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            samples = manifest["samples"]
            actual = {(row["image_id"], row["field"]) for row in samples}
            crops = [row[key] for row in samples for key in ("crop", "raw_crop") if key in row]
            crops.extend(row["crop"] for row in manifest.get("checkboxes", []))
            matches = (actual == expected and len(samples) == len(expected)
                       and (recipe is None or manifest.get("recipe") == recipe)
                       and not manifest.get("errors")
                       and (run / "template_fields.png").is_file()
                       and all((run / crop).is_file() for crop in crops))
            saved = latest_predictions(run)
            if matches and review and saved:
                payload = json.loads(saved["paddle"].read_text(encoding="utf-8"))
                rows = payload["predictions"]
                matches = (payload.get("recipe") == manifest.get("recipe")
                           and len(rows) == len(expected)
                           and {(row["image_id"], row["field"]) for row in rows} == expected)
            if matches and (not review or saved):
                return run
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if review:
        raise FileNotFoundError("Current results do not match. Set RUN_MODE='run' and START_NEW_RUN=True.")
    if run.exists() and not start_new:
        raise ValueError("Current crops do not match. Set START_NEW_RUN=True to rebuild them.")
    return run


def command(project: Path, python: Path, *arguments, check=True) -> subprocess.CompletedProcess:
    """Stream CLI progress into Jupyter and stop the child when the cell is interrupted."""
    arguments = [str(python), "-u", "-m", "card_ocr", *map(str, arguments)]
    with subprocess.Popen(arguments, cwd=project, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1,
                          start_new_session=os.name == "posix") as child:
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
            returncode = child.wait()
        except KeyboardInterrupt:
            if os.name == "posix":
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
                child.wait()
            raise
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, arguments)
    return subprocess.CompletedProcess(arguments, returncode)


def run_paddle(project: Path, python: Path, run: Path, *, rules_path: Path = DEFAULT_RULES) -> Path:
    """Run Paddle on CPU and replace the saved prediction pair after success."""
    manifest = Path(run) / 'manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError('Prepare and review crops before running PaddleOCR')
    output = Path(run) / 'predictions/paddle/results.json'
    result = command(project, python, 'run', '--manifest', manifest, '--output', output, '--rules', rules_path, '--replace',
                     check=False)
    if result.returncode or not output.is_file():
        raise subprocess.CalledProcessError(result.returncode or 1, result.args)
    return output


def latest_predictions(run: Path) -> dict[str, Path]:
    """Return the saved Paddle prediction pair when present."""
    path = Path(run) / 'predictions/paddle/results.json'
    return {'paddle': path} if path.is_file() and path.with_suffix('.csv').is_file() else {}
