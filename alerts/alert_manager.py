"""
alerts/alert_manager.py — Alert Orchestration Layer

Responsibilities:
  - Enforce per-channel cooldowns to prevent alert flooding
  - Dispatch alerts to Email + WhatsApp concurrently
  - Maintain an in-memory alert history (written to JSON for the dashboard)
  - Provide a clean API: alert_manager.trigger(result)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import ALERT_COOLDOWN_SECONDS, LOG_DIR, SCREENSHOT_DIR
from alerts.email_alert import send_email_alert
from alerts.whatsapp_alert import send_whatsapp_alert

logger = logging.getLogger(__name__)

ALERT_LOG_FILE = LOG_DIR / "alert_history.json"


@dataclass
class AlertRecord:
    """Persisted record of a single alert event."""
    id:              int
    timestamp:       str
    detected_class:  str
    confidence:      float
    location:        str
    screenshot_path: Optional[str]
    email_sent:      bool
    whatsapp_sent:   bool


class AlertManager:
    """
    Central alert dispatcher with cooldown enforcement.

    Parameters
    ----------
    location         : Camera / site label shown in alerts
    cooldown_seconds : Minimum time between successive alerts
    enable_email     : Whether to send email notifications
    enable_whatsapp  : Whether to send WhatsApp notifications
    """

    def __init__(
        self,
        location: str = "Camera-01",
        cooldown_seconds: int = ALERT_COOLDOWN_SECONDS,
        enable_email: bool = True,
        enable_whatsapp: bool = True,
    ):
        self.location         = location
        self.cooldown         = cooldown_seconds
        self.enable_email     = enable_email
        self.enable_whatsapp  = enable_whatsapp

        self._last_alert_time: float = 0.0
        self._alert_counter:   int   = 0
        self._history:         List[AlertRecord] = []
        self._lock = threading.Lock()

        self._load_history()

    # ── Public API ───────────────────────────────────────────────────────────

    def trigger(
        self,
        detected_class: str,
        confidence: float,
        frame: Optional[np.ndarray] = None,
        screenshot_path: Optional[Path] = None,
    ) -> bool:
        """
        Attempt to fire an alert.

        Returns True if the alert was dispatched, False if suppressed by cooldown.
        """
        with self._lock:
            now = time.time()
            if now - self._last_alert_time < self.cooldown:
                remaining = self.cooldown - (now - self._last_alert_time)
                logger.debug("Alert suppressed — cooldown %.0fs remaining", remaining)
                return False

            self._last_alert_time = now
            self._alert_counter  += 1
            alert_id = self._alert_counter

        logger.warning(
            "ALERT #%d — %s (%.0f%%) at %s",
            alert_id, detected_class, confidence * 100, self.location,
        )

        # Dispatch in background threads so the video loop isn't blocked
        thread = threading.Thread(
            target=self._dispatch,
            args=(alert_id, detected_class, confidence, screenshot_path),
            daemon=True,
        )
        thread.start()
        return True

    @property
    def history(self) -> List[AlertRecord]:
        return list(reversed(self._history))   # newest first

    @property
    def total_alerts(self) -> int:
        return self._alert_counter

    @property
    def seconds_until_next_alert(self) -> float:
        remaining = self.cooldown - (time.time() - self._last_alert_time)
        return max(0.0, remaining)

    # ── Internal Dispatch ────────────────────────────────────────────────────

    def _dispatch(
        self,
        alert_id: int,
        detected_class: str,
        confidence: float,
        screenshot_path: Optional[Path],
    ) -> None:
        """Run in a background thread: send all configured channels."""
        email_ok     = False
        whatsapp_ok  = False
        ss_str       = str(screenshot_path) if screenshot_path else None

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

        record = AlertRecord(
            id=alert_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            detected_class=detected_class,
            confidence=round(confidence, 4),
            location=self.location,
            screenshot_path=ss_str,
            email_sent=email_ok,
            whatsapp_sent=whatsapp_ok,
        )

        with self._lock:
            self._history.append(record)
            self._save_history()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_history(self) -> None:
        try:
            data = [asdict(r) for r in self._history[-1000:]]   # keep last 1 000 records
            ALERT_LOG_FILE.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.error("Failed to save alert history: %s", exc)

    def _load_history(self) -> None:
        if ALERT_LOG_FILE.exists():
            try:
                data = json.loads(ALERT_LOG_FILE.read_text())
                self._history = [AlertRecord(**r) for r in data]
                self._alert_counter = max((r.id for r in self._history), default=0)
                logger.info("Loaded %d historical alerts.", len(self._history))
            except Exception as exc:
                logger.warning("Could not load alert history: %s", exc)
