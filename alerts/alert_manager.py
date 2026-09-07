"""Local event persistence and optional notification orchestration."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional

from alerts.telegram_alert import send_telegram_alert
from config import (
    ALERT_COOLDOWN_SECONDS,
    ENABLE_TELEGRAM_ALERTS,
    LOG_DIR,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)

logger = logging.getLogger(__name__)
EVENT_LOG_FILE = LOG_DIR / "event_history.json"
LEGACY_ALERT_LOG_FILE = LOG_DIR / "alert_history.json"


@dataclass
class EventRecord:
    id: int
    timestamp: str
    detected_class: str
    confidence: float
    location: str
    source: Optional[str] = None
    screenshot_path: Optional[str] = None
    notification_status: str = "not_attempted"
    notification_channel: Optional[str] = None
    notification_completed_at: Optional[str] = None
    notification_error: Optional[str] = None
    notification_suppression_reason: Optional[str] = None


# Compatibility for callers importing the former name.
AlertRecord = EventRecord


def _legacy_channel(record: dict) -> Optional[str]:
    channels = []
    if record.get("email_sent"):
        channels.append("legacy_email")
    if record.get("whatsapp_sent"):
        channels.append("legacy_whatsapp")
    return ",".join(channels) or None


def _record_from_dict(record: dict) -> EventRecord:
    if "notification_status" in record:
        allowed = {field.name for field in fields(EventRecord)}
        event = EventRecord(
            **{key: value for key, value in record.items() if key in allowed}
        )
        if event.notification_status == "queued":
            event.notification_status = "failed"
            event.notification_completed_at = (
                event.notification_completed_at or event.timestamp
            )
            event.notification_error = (
                "submission outcome unavailable after restart"
            )
        return event

    legacy_status = record.get("status")
    accepted = bool(
        record.get("email_sent")
        or record.get("whatsapp_sent")
        or legacy_status == "delivered"
    )
    if accepted:
        notification_status = "accepted"
        error = record.get("error")
    elif legacy_status == "queued":
        notification_status = "failed"
        error = "delivery outcome unavailable after restart"
    else:
        notification_status = "failed"
        error = record.get("error") or "legacy delivery outcome unavailable"

    return EventRecord(
        id=record["id"],
        timestamp=record["timestamp"],
        detected_class=record["detected_class"],
        confidence=record["confidence"],
        location=record["location"],
        source=record.get("source"),
        screenshot_path=record.get("screenshot_path"),
        notification_status=notification_status,
        notification_channel=_legacy_channel(record),
        notification_completed_at=(
            record.get("completed_at") or record["timestamp"]
        ),
        notification_error=error,
    )


def load_event_history(path: Optional[Path] = None) -> list[EventRecord]:
    history_path = path
    if history_path is None:
        history_path = (
            EVENT_LOG_FILE
            if EVENT_LOG_FILE.exists()
            else LEGACY_ALERT_LOG_FILE
        )
    if not history_path.exists():
        return []
    data = json.loads(history_path.read_text(encoding="utf-8"))
    return [_record_from_dict(record) for record in data]


class AlertManager:
    def __init__(
        self,
        location: str = "Camera-01",
        cooldown_seconds: int = ALERT_COOLDOWN_SECONDS,
        enable_telegram: bool = ENABLE_TELEGRAM_ALERTS,
        telegram_bot_token: str = TELEGRAM_BOT_TOKEN,
        telegram_chat_id: str = TELEGRAM_CHAT_ID,
    ):
        self.location = location
        self.cooldown = max(0, int(cooldown_seconds))
        self.enable_telegram = enable_telegram
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self._last_acceptance_time = 0.0
        self._pending_event_id: Optional[int] = None
        self._event_counter = 0
        self._history: List[EventRecord] = []
        self._lock = threading.Lock()
        self._load_history()

    @property
    def can_notify(self) -> bool:
        with self._lock:
            return self._notification_suppression_reason(time.time()) is None

    def record_event(
        self,
        detected_class: str,
        confidence: float,
        screenshot_path: Optional[Path] = None,
        source: Optional[str] = None,
    ) -> EventRecord:
        """Persist a detected event before considering notification policy."""
        with self._lock:
            self._event_counter += 1
            record = EventRecord(
                id=self._event_counter,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                detected_class=detected_class,
                confidence=round(confidence, 4),
                location=self.location,
                source=source,
                screenshot_path=str(screenshot_path) if screenshot_path else None,
            )
            self._history.append(record)
            try:
                self._save_history()
            except Exception:
                self._history.pop()
                self._event_counter -= 1
                raise
        logger.warning(
            "EVENT #%d — %s (%.0f%%) at %s",
            record.id,
            detected_class,
            confidence * 100,
            self.location,
        )
        return record

    def request_notification(
        self,
        event_id: int,
        detected_class: str,
        confidence: float,
        screenshot_path: Optional[Path] = None,
    ) -> bool:
        """Queue a notification when eligible; never creates the event itself."""
        with self._lock:
            record = self._find_event(event_id)
            reason = self._notification_suppression_reason(time.time())
            if reason is not None:
                record.notification_status = "not_attempted"
                record.notification_suppression_reason = reason
                self._save_history()
                return False

            record.notification_status = "queued"
            record.notification_channel = ",".join(self._enabled_channels())
            record.notification_suppression_reason = None
            self._pending_event_id = event_id
            try:
                self._save_history()
            except Exception:
                record.notification_status = "not_attempted"
                record.notification_channel = None
                self._pending_event_id = None
                raise

        try:
            threading.Thread(
                target=self._dispatch,
                args=(event_id, detected_class, confidence, screenshot_path),
                daemon=True,
            ).start()
        except Exception as exc:
            logger.exception("Could not queue notification for event #%d", event_id)
            with self._lock:
                record.notification_status = "failed"
                record.notification_completed_at = time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                record.notification_error = f"could not queue delivery: {exc}"
                self._pending_event_id = None
                self._save_history()
            return False
        return True

    @property
    def history(self) -> List[EventRecord]:
        with self._lock:
            return list(reversed(self._history))

    @property
    def total_events(self) -> int:
        with self._lock:
            return self._event_counter

    @property
    def accepted_notifications(self) -> int:
        with self._lock:
            return sum(
                record.notification_status == "accepted"
                for record in self._history
            )

    @property
    def seconds_until_next_alert(self) -> float:
        with self._lock:
            return max(
                0.0,
                self.cooldown - (time.time() - self._last_acceptance_time),
            )

    def _enabled_channels(self) -> list[str]:
        return ["telegram"] if self.enable_telegram else []

    def _notification_suppression_reason(self, now: float) -> Optional[str]:
        if self._pending_event_id is not None:
            return "delivery_pending"
        if now - self._last_acceptance_time < self.cooldown:
            return "cooldown"
        if not self.enable_telegram:
            return "telegram_disabled"
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return "telegram_not_configured"
        return None

    def _find_event(self, event_id: int) -> EventRecord:
        try:
            return next(record for record in self._history if record.id == event_id)
        except StopIteration as exc:
            raise ValueError(f"Unknown event id: {event_id}") from exc

    def _dispatch(
        self,
        event_id: int,
        detected_class: str,
        confidence: float,
        screenshot_path: Optional[Path],
    ) -> None:
        try:
            submission = send_telegram_alert(
                bot_token=self.telegram_bot_token,
                chat_id=self.telegram_chat_id,
                detected_class=detected_class,
                confidence=confidence,
                screenshot_path=screenshot_path,
                location=self.location,
            )
        except Exception as exc:
            logger.exception("Notification for event #%d raised an exception", event_id)
            accepted = False
            error = str(exc)
        else:
            accepted = submission.accepted
            error = submission.error
        completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            record = self._find_event(event_id)
            record.notification_status = "accepted" if accepted else "failed"
            record.notification_completed_at = completed_at
            record.notification_error = error
            if accepted:
                self._last_acceptance_time = time.time()
            if self._pending_event_id == event_id:
                self._pending_event_id = None
            self._save_history()

    def _save_history(self) -> None:
        data = [asdict(record) for record in self._history[-1000:]]
        partial = EVENT_LOG_FILE.with_name(EVENT_LOG_FILE.name + ".part")
        try:
            partial.write_text(json.dumps(data, indent=2), encoding="utf-8")
            partial.replace(EVENT_LOG_FILE)
        except Exception as exc:
            partial.unlink(missing_ok=True)
            logger.exception("Failed to save event history")
            raise RuntimeError(f"Could not persist event history: {exc}") from exc

    def _load_history(self) -> None:
        if not EVENT_LOG_FILE.exists() and not LEGACY_ALERT_LOG_FILE.exists():
            return
        try:
            self._history = load_event_history()
            self._event_counter = max(
                (record.id for record in self._history),
                default=0,
            )
        except Exception as exc:
            logger.warning("Could not load event history: %s", exc)
