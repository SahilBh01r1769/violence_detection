"""
core/detector.py — YOLOv8 Violence Detection Engine

Responsibilities:
  - Load and cache the YOLOv8 model
  - Run inference on individual frames
  - Apply temporal consistency check (N consecutive frames)
  - Return structured DetectionResult objects
"""

from __future__ import annotations

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single bounding-box detection from one frame."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2


@dataclass
class DetectionResult:
    """Aggregated result for one processed frame."""
    frame: np.ndarray
    detections: List[Detection] = field(default_factory=list)
    is_violent: bool = False
    alert_triggered: bool = False   # True only when consistency threshold met
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0

    @property
    def primary_class(self) -> str:
        if not self.detections:
            return "Normal"
        return max(self.detections, key=lambda d: d.confidence).class_name

    @property
    def max_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return max(d.confidence for d in self.detections)


class ViolenceDetector:
    """
    Wraps a YOLOv8 model and adds temporal consistency logic.

    Parameters
    ----------
    model_path        : Path to .pt weights file
    confidence        : Minimum confidence threshold (0–1)
    frame_consistency : Number of consecutive violent frames before alert fires
    violence_classes  : List of class names treated as violent
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.55,
        frame_consistency: int = 5,
        violence_classes: Optional[List[str]] = None,
    ):
        self.confidence       = confidence
        self.frame_consistency = frame_consistency
        self.violence_classes = set(violence_classes or ["Fighting", "Weapon", "Aggression"])

        self._model       = None
        self._model_path  = model_path
        self._frame_id    = 0

        # Sliding window — True = violent, False = safe
        self._window: deque[bool] = deque(maxlen=frame_consistency)

        self._load_model()

    # ── Model Loading ────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load YOLOv8 weights. Falls back to yolov8n.pt if file missing."""
        try:
            from ultralytics import YOLO
            path = Path(self._model_path)
            if not path.exists():
                logger.warning(
                    "Model not found at '%s'. "
                    "Using pretrained yolov8n.pt — replace with your fine-tuned weights.",
                    self._model_path,
                )
                self._model = YOLO("yolov8n.pt")
            else:
                self._model = YOLO(str(path))
                logger.info("Loaded model from %s", path)
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

    # ── Inference ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference on a single BGR frame.

        Returns a DetectionResult; alert_triggered is True only when the
        temporal consistency window is fully saturated with violent frames.
        """
        self._frame_id += 1
        result = DetectionResult(frame=frame.copy(), frame_id=self._frame_id)

        try:
            predictions = self._model.predict(
                source=frame,
                conf=self.confidence,
                verbose=False,
            )

            for pred in predictions:
                if pred.boxes is None:
                    continue
                for box in pred.boxes:
                    cls_id     = int(box.cls[0])
                    class_name = self._model.names.get(cls_id, str(cls_id))
                    conf       = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    result.detections.append(
                        Detection(
                            class_name=class_name,
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                        )
                    )
                    if class_name in self.violence_classes:
                        result.is_violent = True

        except Exception as exc:
            logger.error("Inference error on frame %d: %s", self._frame_id, exc)

        # ── Temporal consistency ─────────────────────────────────────────────
        self._window.append(result.is_violent)

        if (
            len(self._window) == self.frame_consistency
            and all(self._window)
        ):
            result.alert_triggered = True
            # Reset window so we don't fire on every subsequent frame
            self._window.clear()

        return result

    # ── Annotated Frame ──────────────────────────────────────────────────────

    @staticmethod
    def annotate_frame(result: DetectionResult) -> np.ndarray:
        """
        Draw bounding boxes and labels on a copy of the frame.
        Returns an annotated BGR image.
        """
        frame = result.frame.copy()
        h, w  = frame.shape[:2]

        # Status banner
        if result.alert_triggered:
            banner_color = (0, 0, 220)
            label        = "⚠ VIOLENCE DETECTED"
        elif result.is_violent:
            banner_color = (0, 100, 255)
            label        = "SUSPICIOUS ACTIVITY"
        else:
            banner_color = (30, 160, 30)
            label        = "MONITORING"

        cv2.rectangle(frame, (0, 0), (w, 36), banner_color, -1)
        cv2.putText(
            frame, label,
            (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2,
        )

        # Timestamp
        ts = time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime(result.timestamp))
        cv2.putText(
            frame, ts,
            (w - 230, 25), cv2.FONT_HERSHEY_PLAIN, 1.1, (220, 220, 220), 1,
        )

        # Bounding boxes
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            is_violent = det.class_name in {"Fighting", "Weapon", "Aggression"}
            colour = (0, 0, 255) if is_violent else (0, 200, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            box_label = f"{det.class_name}  {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
            cv2.putText(
                frame, box_label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )

        return frame

    # ── Stats ────────────────────────────────────────────────────────────────

    @property
    def model_classes(self) -> dict:
        return self._model.names if self._model else {}
