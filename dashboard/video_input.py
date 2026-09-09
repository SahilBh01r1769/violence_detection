"""Safe local storage for videos selected through the dashboard."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "runtime_uploads"
ALLOWED_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def persist_uploaded_video(
    filename: str,
    content: bytes,
    directory: Path = UPLOAD_DIR,
) -> Path:
    """Persist an uploaded video under a content-addressed safe filename."""
    original = Path(filename).name
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError("unsupported video type")
    if not content:
        raise ValueError("uploaded video is empty")

    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(original).stem).strip("-")
    stem = stem or "video"
    digest = hashlib.sha256(content).hexdigest()[:12]
    destination = Path(directory) / f"{stem}-{digest}{suffix}"
    if destination.exists() and destination.stat().st_size == len(content):
        return destination.resolve()

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        partial.write_bytes(content)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination.resolve()
