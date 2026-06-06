"""
core/stream.py — Video Stream Manager

Handles:
  - Opening webcam / RTSP / file sources
  - Controlled-rate frame reading (FPS throttle)
  - Screenshot saving with timestamped filenames
  - Auto-cleanup of old screenshots
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

from config import SCREENSHOT_DIR, MAX_SCREENSHOTS, FPS_TARGET

logger = logging.getLogger(__name__)


class VideoStream:
    """
    Context-manager wrapper around OpenCV VideoCapture.

    Usage
    -----
    with VideoStream(source=0) as stream:
        for frame in stream.frames():
            ...
    """

    def __init__(
        self,
        source=0,
        fps_target: int = FPS_TARGET,
        screenshot_dir: Path = SCREENSHOT_DIR,
    ):
        self.source         = source
        self.fps_target     = fps_target
        self.screenshot_dir = Path(screenshot_dir)
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count   = 0
        self._start_time    = time.time()

    # ── Context Manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "VideoStream":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source: {self.source!r}. "
                "Check that your webcam is connected or the RTSP URL is correct."
            )
        # Suggest native resolution; camera may override
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        logger.info("Opened video source: %s", self.source)

    def release(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Video source released.")

    # ── Frame Generator ──────────────────────────────────────────────────────

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Yield BGR frames at approximately fps_target FPS.
        Handles dropped frames gracefully.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Call open() before iterating frames.")

        frame_interval = 1.0 / self.fps_target
        last_time      = time.time()

        while True:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Frame read failed — attempting reconnect…")
                time.sleep(0.5)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind for file sources
                ret, frame = self._cap.read()
                if not ret:
                    logger.error("Cannot recover stream. Stopping.")
                    break

            self._frame_count += 1

            # Throttle to target FPS
            elapsed   = time.time() - last_time
            wait_time = frame_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            last_time = time.time()

            yield frame

    # ── Screenshot ───────────────────────────────────────────────────────────

    def save_screenshot(
        self,
        frame: np.ndarray,
        prefix: str = "alert",
    ) -> Path:
        """
        Save *frame* as a timestamped JPEG inside screenshot_dir.
        Auto-purges oldest files when MAX_SCREENSHOTS is exceeded.
        Returns the saved file path.
        """
        ts       = time.strftime("%Y%m%d_%H%M%S")
        filename = self.screenshot_dir / f"{prefix}_{ts}_{self._frame_count:06d}.jpg"
        cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        logger.info("Screenshot saved: %s", filename)

        self._purge_old_screenshots()
        return filename

    def _purge_old_screenshots(self) -> None:
        files = sorted(self.screenshot_dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime)
        while len(files) > MAX_SCREENSHOTS:
            files.pop(0).unlink(missing_ok=True)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def actual_fps(self) -> float:
        elapsed = self.uptime_seconds
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    @property
    def resolution(self) -> tuple[int, int]:
        if self._cap:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0
