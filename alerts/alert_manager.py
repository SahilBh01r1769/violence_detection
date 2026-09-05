"""Alert orchestration, cooldown enforcement and alert-history persistence."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import ALERT_COOLDOWN_SECONDS, LOG_DIR
from alerts.email_alert import send_email_alert
from alerts.whatsapp_alert import send_whatsapp_alert

logger = logging.getLogger(__name__)
ALERT_LOG_FILE = LOG_DIR / "alert_history.json"


@dataclass
class AlertRecord:
    id: int
    timestamp: str
    detected_class: str
    confidence: float
    location: str
    screenshot_path: Optional[str]
    email_sent: bool = False
    whatsapp_sent: bool = False
    status: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status is None:
            delivered = self.email_sent or self.whatsapp_sent
            self.status = "delivered" if delivered else "failed"
            self.completed_at = self.completed_at or self.timestamp
            if not delivered and self.error is None:
                self.error = "delivery outcome unavailable"


class AlertManager:
    def __init__(self, location: str = "Camera-01", cooldown_seconds: int = ALERT_COOLDOWN_SECONDS, enable_email: bool = True, enable_whatsapp: bool = True):
        self.location = location
        self.cooldown = max(0, int(cooldown_seconds))
        self.enable_email = enable_email
        self.enable_whatsapp = enable_whatsapp
        self._last_delivery_time = 0.0
        self._pending_alert_id: Optional[int] = None
        self._alert_counter = 0
        self._history: List[AlertRecord] = []
        self._lock = threading.Lock()
        self._load_history()

    @property
    def can_trigger(self) -> bool:
        with self._lock:
            return (
                self._pending_alert_id is None
                and time.time() - self._last_delivery_time >= self.cooldown
            )

    def trigger(self, detected_class: str, confidence: float, frame: Optional[np.ndarray] = None, screenshot_path: Optional[Path] = None) -> bool:
        with self._lock:
            now = time.time()
            if self._pending_alert_id is not None:
                return False
            if now - self._last_delivery_time < self.cooldown:
                return False
            self._alert_counter += 1
            alert_id = self._alert_counter
            record = AlertRecord(
                alert_id,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                detected_class,
                round(confidence, 4),
                self.location,
                str(screenshot_path) if screenshot_path else None,
                status="queued",
            )
            self._history.append(record)
            self._pending_alert_id = alert_id
            self._save_history()

        logger.warning("ALERT #%d — %s (%.0f%%) at %s", alert_id, detected_class, confidence * 100, self.location)
        try:
            threading.Thread(target=self._dispatch, args=(alert_id, detected_class, confidence, screenshot_path), daemon=True).start()
        except Exception as exc:
            logger.exception("Could not queue alert #%d", alert_id)
            with self._lock:
                record.status = "failed"
                record.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
                record.error = f"could not queue delivery: {exc}"
                self._pending_alert_id = None
                self._save_history()
            return False
        return True

    @property
    def history(self) -> List[AlertRecord]:
        return list(reversed(self._history))

    @property
    def total_alerts(self) -> int:
        return self._alert_counter

    @property
    def seconds_until_next_alert(self) -> float:
        with self._lock:
            return max(0.0, self.cooldown - (time.time() - self._last_delivery_time))

    def _dispatch(self, alert_id: int, detected_class: str, confidence: float, screenshot_path: Optional[Path]) -> None:
        email_ok = False
        whatsapp_ok = False
        error: Optional[str] = None
        try:
            if self.enable_email:
                email_ok = send_email_alert(
                    detected_class=detected_class,
                    confidence=confidence,
                    screenshot_path=screenshot_path,
                    location=self.location,
                )
            if self.enable_whatsapp:
                whatsapp_ok = send_whatsapp_alert(
                    detected_class=detected_class,
                    confidence=confidence,
                    screenshot_path=screenshot_path,
                    location=self.location,
                )
        except Exception as exc:
            logger.exception("Alert #%d delivery raised an exception", alert_id)
            error = str(exc)

        delivered = email_ok or whatsapp_ok
        if not delivered and error is None:
            error = (
                "no delivery channels enabled"
                if not self.enable_email and not self.enable_whatsapp
                else "all enabled delivery channels failed"
            )
        completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            record = next(record for record in self._history if record.id == alert_id)
            record.email_sent = email_ok
            record.whatsapp_sent = whatsapp_ok
            record.status = "delivered" if delivered else "failed"
            record.completed_at = completed_at
            record.error = error
            if delivered:
                self._last_delivery_time = time.time()
            if self._pending_alert_id == alert_id:
                self._pending_alert_id = None
            self._save_history()

    def _save_history(self) -> None:
        try:
            data = [asdict(record) for record in self._history[-1000:]]
            ALERT_LOG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to save alert history: %s", exc)

    def _load_history(self) -> None:
        if not ALERT_LOG_FILE.exists():
            return
        try:
            data = json.loads(ALERT_LOG_FILE.read_text(encoding="utf-8"))
            self._history = [AlertRecord(**record) for record in data]
            self._alert_counter = max((record.id for record in self._history), default=0)
        except Exception as exc:
            logger.warning("Could not load alert history: %s", exc)
