"""Select scan IDs for crop preparation and notebook review."""

from pathlib import Path
import re


def select_images(data_dir: Path, *, ids=None, count=50, require_count=True) -> list[str]:
    """Normalize numeric IDs and reject missing, duplicate, or unsafe selections."""
    if count < 1:
        raise ValueError("Sample count must be positive")
    available = sorted(path.stem for path in Path(data_dir).iterdir()
                       if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    if len(available) != len(set(available)):
        raise ValueError("Multiple image files share the same ID")
    selected = available[:count] if ids is None else [
        item.zfill(4) if item not in available and item.isdigit() else item for item in ids]
    if (not selected or len(selected) != len(set(selected))
            or any(item not in available for item in selected)
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", item) for item in selected)):
        raise ValueError("Choose unique, existing image IDs with letters, digits, underscores, or hyphens")
    if ids is None and require_count and len(selected) < count:
        raise ValueError(f"Requested {count} images, but only {len(selected)} are available; lower the sample count")
    return selected
