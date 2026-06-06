"""
core/pipeline.py — Main Detection Pipeline

Ties together:
  VideoStream → ViolenceDetector → AlertManager → Screenshot

Run this directly for headless operation, or import run_pipeline()
for use inside the Streamlit dashboard.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from config import (
    ALERT_COOLDOWN_SECONDS,
    CONFIDENCE_THRESHOLD,
    FRAME_CONSISTENCY,
    MODEL_PATH,
    VIDEO_SOURCE,
    VIOLENCE_CLASSES,
)
from core.detector import ViolenceDetector
from core.stream import VideoStream
from alerts.alert_manager import AlertManager

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """
    Full detection pipeline.

    Parameters
    ----------
    on_frame   : Optional callback(annotated_frame, result) for custom handling
                 (e.g., feeding frames to Streamlit)
    location   : Camera label used in alerts
    show_window: Display an OpenCV window (useful for local testing)
    """

    def __init__(
        self,
        on_frame: Optional[Callable] = None,
        location: str = "Camera-01",
        show_window: bool = False,
    ):
        self.on_frame    = on_frame
        self.location    = location
        self.show_window = show_window
        self._running    = False

        self.detector = ViolenceDetector(
            model_path=MODEL_PATH,
            confidence=CONFIDENCE_THRESHOLD,
            frame_consistency=FRAME_CONSISTENCY,
            violence_classes=VIOLENCE_CLASSES,
        )

        self.alert_manager = AlertManager(
            location=location,
            cooldown_seconds=ALERT_COOLDOWN_SECONDS,
            enable_email=True,
            enable_whatsapp=True,
        )

        # Stats
        self.frames_processed: int   = 0
        self.alerts_fired:     int   = 0
        self.start_time:       float = 0.0

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self, source=None) -> None:
        """
        Start the pipeline. Blocks until stopped or an error occurs.
        Press Q (if show_window=True) or send SIGINT to stop.
        """
        source = source or VIDEO_SOURCE
        self._running    = True
        self.start_time  = time.time()

        # Graceful shutdown on Ctrl-C
        signal.signal(signal.SIGINT,  self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

        logger.info("Starting detection pipeline on source: %s", source)

        try:
            with VideoStream(source=source) as stream:
                for frame in stream.frames():
                    if not self._running:
                        break

                    result = self.detector.process_frame(frame)
                    self.frames_processed += 1

                    annotated = ViolenceDetector.annotate_frame(result)

                    # ── Alert + Screenshot ───────────────────────────────────
                    if result.alert_triggered:
                        ss_path = stream.save_screenshot(annotated, prefix="alert")
                        fired   = self.alert_manager.trigger(
                            detected_class=result.primary_class,
                            confidence=result.max_confidence,
                            frame=annotated,
                            screenshot_path=ss_path,
                        )
                        if fired:
                            self.alerts_fired += 1

                    # ── Custom Callback ──────────────────────────────────────
                    if self.on_frame:
                        self.on_frame(annotated, result)

                    # ── Optional Display ─────────────────────────────────────
                    if self.show_window:
                        cv2.imshow("Violence Detection System", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

        except RuntimeError as exc:
            logger.error("Pipeline error: %s", exc)
        finally:
            if self.show_window:
                cv2.destroyAllWindows()
            logger.info(
                "Pipeline stopped. Frames: %d | Alerts: %d | Uptime: %.0fs",
                self.frames_processed,
                self.alerts_fired,
                time.time() - self.start_time,
            )

    def stop(self) -> None:
        self._running = False

    def _handle_stop(self, *_) -> None:
        logger.info("Shutdown signal received.")
        self._running = False

    # ── Stats ────────────────────────────────────────────────────────────────

    @property
    def uptime(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0

    @property
    def fps(self) -> float:
        return self.frames_processed / self.uptime if self.uptime > 0 else 0.0


# ── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser(description="Violence Detection Pipeline")
    parser.add_argument("--source",   default=None,  help="Video source (0, RTSP URL, or file path)")
    parser.add_argument("--location", default="Camera-01", help="Camera label for alerts")
    parser.add_argument("--display",  action="store_true",  help="Show OpenCV window")
    args = parser.parse_args()

    pipeline = DetectionPipeline(
        location=args.location,
        show_window=args.display,
    )
    pipeline.run(source=args.source)
