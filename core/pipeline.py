"""Main video -> detection -> alert pipeline."""

from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import cv2

from alerts.alert_manager import AlertManager
from config import ALERT_COOLDOWN_SECONDS, CONFIDENCE_THRESHOLD, ENABLE_EMAIL_ALERTS, ENABLE_WHATSAPP_ALERTS, FRAME_CONSISTENCY, MODEL_PATH, VIDEO_SOURCE, VIOLENCE_CLASSES, VIOLENCE_CLASS_IDS
from core.detector import ViolenceDetector
from core.stream import VideoStream, safe_source_label

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeFailure:
    stage: str
    message: str
    timestamp: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def normalise_source(source):
    if source is None:
        return VIDEO_SOURCE
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


class DetectionPipeline:
    def __init__(self, on_frame: Optional[Callable] = None, location: str = "Camera-01", show_window: bool = False, confidence: float = CONFIDENCE_THRESHOLD, frame_consistency: int = FRAME_CONSISTENCY, cooldown_seconds: int = ALERT_COOLDOWN_SECONDS, enable_email: bool = ENABLE_EMAIL_ALERTS, enable_whatsapp: bool = ENABLE_WHATSAPP_ALERTS):
        self.on_frame = on_frame
        self.location = location
        self.show_window = show_window
        self._running = False
        self.detector = ViolenceDetector(MODEL_PATH, confidence, frame_consistency, VIOLENCE_CLASSES, VIOLENCE_CLASS_IDS)
        self.alert_manager = AlertManager(location, cooldown_seconds, enable_email, enable_whatsapp)
        self.frames_processed = 0
        self.alerts_fired = 0
        self.start_time = 0.0
        self.end_time = 0.0
        self.source_state = "idle"
        self.last_error: Optional[RuntimeFailure] = None

    def run(self, source=None) -> None:
        source = normalise_source(source)
        self.frames_processed = 0
        self.alerts_fired = 0
        self.start_time = time.time()
        self.end_time = 0.0
        self.source_state = "connecting"
        self.last_error = None
        self._running = True
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._handle_stop)
            signal.signal(signal.SIGTERM, self._handle_stop)
        logger.info("Starting detection pipeline on source: %s", safe_source_label(source))
        try:
            with VideoStream(source=source) as stream:
                self.source_state = "connected"
                for frame in stream.frames():
                    if not self._running:
                        break
                    try:
                        result = self.detector.process_frame(frame)
                    except Exception as exc:
                        self._record_error("inference", exc)
                        raise
                    self.frames_processed += 1
                    annotated = ViolenceDetector.annotate_frame(result)
                    if result.alert_triggered and self.alert_manager.can_trigger:
                        try:
                            screenshot_path = stream.save_screenshot(annotated, prefix="alert")
                            if self.alert_manager.trigger(result.primary_class, result.max_confidence, annotated, screenshot_path):
                                self.alerts_fired += 1
                        except Exception as exc:
                            self._record_error("alert", exc)
                            raise
                    if self.on_frame:
                        self.on_frame(annotated, result)
                    if self.show_window:
                        cv2.imshow("Violence Detection System", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                if stream.end_reason == "disconnected":
                    self.source_state = "disconnected"
                    self._record_error("source", "video source disconnected")
                elif not self._running:
                    self.source_state = "stopped"
                else:
                    self.source_state = stream.end_reason or "ended"
        except Exception as exc:
            if self.last_error is None:
                stage = "source" if self.source_state == "connecting" else "pipeline"
                self._record_error(stage, exc)
            self.source_state = "error"
            logger.exception("Pipeline stopped because of an error: %s", exc)
        finally:
            self._running = False
            self.end_time = time.time()
            if self.show_window:
                cv2.destroyAllWindows()
            logger.info("Pipeline stopped. Frames: %d | Alerts: %d | Uptime: %.0fs", self.frames_processed, self.alerts_fired, self.uptime)

    def stop(self) -> None:
        self._running = False

    def _handle_stop(self, *_args) -> None:
        logger.info("Shutdown signal received")
        self.stop()

    def _record_error(self, stage: str, error: Exception | str) -> None:
        self.last_error = RuntimeFailure(
            stage=stage,
            message=str(error),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @property
    def uptime(self) -> float:
        if not self.start_time:
            return 0.0
        endpoint = time.time() if self._running else (self.end_time or time.time())
        return max(0.0, endpoint - self.start_time)

    @property
    def fps(self) -> float:
        return self.frames_processed / self.uptime if self.uptime > 0 else 0.0


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
    parser = argparse.ArgumentParser(description="Violence Detection Pipeline")
    parser.add_argument("--source", default=None, help="0 for webcam, RTSP URL, or video-file path")
    parser.add_argument("--location", default="Camera-01", help="Camera label for alerts")
    parser.add_argument("--display", action="store_true", help="Show OpenCV window")
    args = parser.parse_args()
    DetectionPipeline(location=args.location, show_window=args.display).run(source=args.source)
