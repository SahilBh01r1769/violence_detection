"""Download the configured pretrained violence-detection checkpoint."""

from __future__ import annotations

import argparse
import logging
import shutil
import urllib.request
from pathlib import Path

from config import MODEL_DOWNLOAD_URL, MODEL_PATH

logger = logging.getLogger(__name__)


def ensure_model(path: Path = MODEL_PATH, url: str = MODEL_DOWNLOAD_URL, force: bool = False) -> Path:
    path = Path(path)
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    if not url:
        raise RuntimeError(f"Model not found at {path} and MODEL_DOWNLOAD_URL is empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")
    logger.info("Downloading pretrained violence model to %s", path)

    request = urllib.request.Request(url, headers={"User-Agent": "violence-detection-project/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temp_path.stat().st_size == 0:
            raise RuntimeError("Downloaded model file is empty")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the configured pretrained violence model")
    parser.add_argument("--force", action="store_true", help="Re-download even if the model already exists")
    args = parser.parse_args()
    downloaded = ensure_model(force=args.force)
    print(downloaded)
