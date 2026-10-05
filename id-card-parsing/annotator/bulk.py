"""Bulk-save ID card annotations using independent template detections.

Selects available images within an inclusive numeric ID range. Existing
annotations are preserved, while missing images, unreliable detections,
and processing failures are reported as skipped or failed.

Processing runs sequentially and provides progress snapshots with per-image
results. Cancellation stops further processing when observed and retains
annotations already saved. Bulk saving does not copy the current editor box
or its manual edits to other images.
"""

from dataclasses import dataclass, field, replace


def ids_in_range(image_ids, start: str, end: str) -> tuple[list[str], int]:
    """Return available numeric image IDs within an inclusive range and the
    number of missing IDs. Raise ValueError for invalid or reversed bounds.
    """
    start, end = start.strip(), end.strip()
    if not all(value.isascii() and value.isdecimal() for value in (start, end)):
        raise ValueError("Enter numeric start and end image IDs.")
    low, high = int(start), int(end)
    if low > high:
        raise ValueError("Start ID must be less than or equal to end ID.")
    selected = [image_id for image_id in image_ids
                if image_id.isascii() and image_id.isdecimal()
                and low <= int(image_id) <= high]
    selected.sort(key=lambda value: (int(value), value))
    missing = high-low+1-len({int(value) for value in selected})
    return selected, missing


@dataclass
class BulkSummary:
    """Track batch progress, outcome counts, cancellation, and per-image results."""
    total: int
    completed: int = 0
    saved: int = 0
    existing: int = 0
    unmatched: int = 0
    failed: int = 0
    cancelled: bool = False
    current_id: str = ""
    last_error: str = ""
    outcomes: list[tuple[str, str, str]] = field(default_factory=list)


def save_template_range(backend, matcher, image_ids, cancel, on_progress) -> BulkSummary:
    """Detect and save each image independently without overwriting annotations.

    Report progress after each completed image. Continue after per-image failures,
    stop when cancellation is observed, and retain files already committed.
    """
    summary = BulkSummary(total=len(image_ids))
    for image_id in image_ids:
        if cancel.is_set():
            break
        summary.current_id = image_id
        path = backend.config.out_dir / f"{image_id}.json"
        try:
            if path.exists():
                summary.existing += 1
                summary.outcomes.append((image_id, "Skipped", "Annotation already exists"))
            else:
                preview, _ = backend.open(image_id)
                if cancel.is_set():
                    break
                match = matcher.match(preview.preview)
                if cancel.is_set():
                    break
                if match is None:
                    summary.unmatched += 1
                    summary.outcomes.append((image_id, "Skipped", "No reliable detection"))
                elif path.exists():
                    summary.existing += 1
                    summary.outcomes.append((image_id, "Skipped", "Annotation already exists"))
                else:
                    backend.export(image_id, preview, match.obb, overwrite=False, cancel=cancel)
                    summary.saved += 1
                    summary.outcomes.append((image_id, "Saved", "Automatic box saved"))
        except FileExistsError:
            summary.existing += 1
            summary.outcomes.append((image_id, "Skipped", "Annotation already exists"))
        except InterruptedError as exc:
            if cancel.is_set():
                break
            summary.failed += 1
            summary.last_error = f"{image_id}: {exc}"
            summary.outcomes.append((image_id, "Failed", str(exc)))
        except Exception as exc:
            summary.failed += 1
            summary.last_error = f"{image_id}: {exc}"
            summary.outcomes.append((image_id, "Failed", str(exc)))
        summary.completed += 1
        on_progress(replace(summary, outcomes=list(summary.outcomes)))
    summary.cancelled = cancel.is_set()
    return summary
