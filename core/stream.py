"""Video stream manager for webcams, RTSP streams and video files."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Generator, Optional
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from config import FPS_TARGET, MAX_SCREENSHOTS, SCREENSHOT_DIR

logger = logging.getLogger(__name__)


def safe_source_label(source) -> str:
    text = str(source)
    if "://" not in text:
        return text
    try:
        parts = urlsplit(text)
        if parts.username is None:
            return text
        host = parts.hostname or ""
        if parts.port:
            host += f":{parts.port}"
        user = parts.username
        netloc = f"{user}:***@{host}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return "<network stream>"


class VideoStream:
    def __init__(self, source=0, fps_target: int = FPS_TARGET, screenshot_dir: Path = SCREENSHOT_DIR):
        self.source = source
        self.fps_target = max(1, int(fps_target))
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._start_time = time.time()
        self._is_file = isinstance(source, str) and Path(source).is_file()

    def __enter__(self) -> "VideoStream":
        self.open()
        return self

    def __exit__(self, *_args) -> None:
        self.release()

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {safe_source_label(self.source)!r}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        logger.info("Opened video source: %s", safe_source_label(self.source))

    def release(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Video source released")

    def frames(self) -> Generator[np.ndarray, None, None]:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Call open() before iterating frames")
        frame_interval = 1.0 / self.fps_target
        last_time = time.time()
        while True:
            ret, frame = self._cap.read()
            if not ret:
                if self._is_file:
                    logger.info("Reached end of video file")
                    break
                logger.warning("Frame read failed; attempting one reconnect")
                time.sleep(0.5)
                self.release()
                self._cap = cv2.VideoCapture(self.source)
                ret, frame = self._cap.read() if self._cap.isOpened() else (False, None)
                if not ret:
                    logger.error("Cannot recover stream; stopping")
                    break
            self._frame_count += 1
            wait_time = frame_interval - (time.time() - last_time)
            if wait_time > 0:
                time.sleep(wait_time)
            last_time = time.time()
            yield frame

    def save_screenshot(self, frame: np.ndarray, prefix: str = "alert") -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = self.screenshot_dir / f"{prefix}_{ts}_{self._frame_count:06d}.jpg"
        if not cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"Could not save screenshot to {filename}")
        self._purge_old_screenshots()
        return filename

    def _purge_old_screenshots(self) -> None:
        files = sorted(self.screenshot_dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime)
        while len(files) > MAX_SCREENSHOTS:
            files.pop(0).unlink(missing_ok=True)
