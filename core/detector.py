"""YOLOv8 violence detector with an N-frame temporal consistency filter."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import AUTO_DOWNLOAD_MODEL, MODEL_DOWNLOAD_URL, VIOLENCE_CLASS_IDS
from utils.download_model import ensure_model

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    is_violent: bool = False


@dataclass
class DetectionResult:
    frame: np.ndarray
    detections: List[Detection] = field(default_factory=list)
    is_violent: bool = False
    alert_triggered: bool = False
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0

    @property
    def violent_detections(self) -> List[Detection]:
        return [d for d in self.detections if d.is_violent]

    @property
    def primary_class(self) -> str:
        candidates = self.violent_detections or self.detections
        return max(candidates, key=lambda d: d.confidence).class_name if candidates else "Normal"

    @property
    def max_confidence(self) -> float:
        candidates = self.violent_detections or self.detections
        return max((d.confidence for d in candidates), default=0.0)


class ViolenceDetector:
    def __init__(self, model_path: str | Path, confidence: float = 0.55, frame_consistency: int = 5, violence_classes: Optional[List[str]] = None, violence_class_ids: Optional[set[int]] = None):
        self.confidence = confidence
        self.frame_consistency = max(1, int(frame_consistency))
        self.violence_classes = {self._normalise_name(name) for name in (violence_classes or ["violence", "fight", "fighting", "violence/fight"])}
        self.violence_class_ids = set(VIOLENCE_CLASS_IDS if violence_class_ids is None else violence_class_ids)
        self._model = None
        self._model_path = Path(model_path)
        self._frame_id = 0
        self._window: deque[bool] = deque(maxlen=self.frame_consistency)
        self._load_model()

    @staticmethod
    def _normalise_name(name: str) -> str:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    def _is_violent_class(self, class_id: int, class_name: str) -> bool:
        return class_id in self.violence_class_ids or self._normalise_name(class_name) in self.violence_classes

    def _class_name(self, class_id: int) -> str:
        names = self._model.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed; run pip install -r requirements.txt") from exc
        path = self._model_path
        if not path.exists():
            if not AUTO_DOWNLOAD_MODEL:
                raise RuntimeError(f"Violence model not found at {path}. Run: python -m utils.download_model")
            try:
                path = ensure_model(path=path, url=MODEL_DOWNLOAD_URL)
            except Exception as exc:
                raise RuntimeError(f"Could not download violence model to {path}. Run python -m utils.download_model when internet access is available.") from exc
        self._model = YOLO(str(path))
        logger.info("Loaded violence model from %s with classes %s", path, self._model.names)

    def set_frame_consistency(self, value: int) -> None:
        self.frame_consistency = max(1, int(value))
        self._window = deque(maxlen=self.frame_consistency)

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        self._frame_id += 1
        result = DetectionResult(frame=frame.copy(), frame_id=self._frame_id)
        try:
            predictions = self._model.predict(source=frame, conf=self.confidence, verbose=False)
            for pred in predictions:
                if pred.boxes is None:
                    continue
                for box in pred.boxes:
                    class_id = int(box.cls[0])
                    class_name = self._class_name(class_id)
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    is_violent = self._is_violent_class(class_id, class_name)
                    result.detections.append(Detection(class_id, class_name, confidence, (x1, y1, x2, y2), is_violent))
                    result.is_violent = result.is_violent or is_violent
        except Exception as exc:
            logger.exception("Inference failed on frame %d", self._frame_id)
            raise RuntimeError(f"Model inference failed on frame {self._frame_id}") from exc

        self._window.append(result.is_violent)
        if len(self._window) == self.frame_consistency and all(self._window):
            result.alert_triggered = True
            self._window.clear()
        return result

    @staticmethod
    def annotate_frame(result: DetectionResult) -> np.ndarray:
        frame = result.frame.copy()
        _, width = frame.shape[:2]
        if result.alert_triggered:
            banner_color, label = (0, 0, 220), "VIOLENCE DETECTED"
        elif result.is_violent:
            banner_color, label = (0, 100, 255), "SUSPICIOUS ACTIVITY"
        else:
            banner_color, label = (30, 160, 30), "MONITORING"
        cv2.rectangle(frame, (0, 0), (width, 36), banner_color, -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        timestamp = time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime(result.timestamp))
        cv2.putText(frame, timestamp, (max(10, width - 230), 25), cv2.FONT_HERSHEY_PLAIN, 1.1, (220, 220, 220), 1)
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            colour = (0, 0, 255) if det.is_violent else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            box_label = f"{det.class_name}  {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            top = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, top), (x1 + tw + 6, y1), colour, -1)
            cv2.putText(frame, box_label, (x1 + 3, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return frame

    @property
    def model_classes(self) -> dict | list:
        return self._model.names if self._model else {}
