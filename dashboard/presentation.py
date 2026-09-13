"""Small, framework-independent presentation helpers for the dashboard."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import PureWindowsPath


def compact_source_label(source: object) -> str:
    """Return a readable source name without exposing a local directory."""
    if source is None or str(source).strip() == "":
        return "Unknown"
    text = str(source)
    if "://" in text:
        return "Network stream"
    if "\\" in text:
        return PureWindowsPath(text).name or text
    return text.rsplit("/", 1)[-1]


def notification_label(event: dict) -> str:
    status = str(event.get("notification_status") or "not_attempted")
    labels = {
        "not_attempted": "Not attempted",
        "queued": "Queued",
        "accepted": "Accepted",
        "failed": "Failed",
    }
    label = labels.get(status, status.replace("_", " ").title())
    reason = event.get("notification_suppression_reason")
    if status == "not_attempted" and reason:
        label += f" — {str(reason).replace('_', ' ')}"
    return label


def format_duration(seconds: object) -> str:
    """Format runtime seconds without suggesting more precision than we have."""
    try:
        total = max(0, int(float(seconds or 0)))
    except (TypeError, ValueError):
        total = 0
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def runtime_state_label(status: dict) -> tuple[str, str]:
    """Return a concise state label and semantic CSS class."""
    if status.get("last_error") or status.get("source_state") == "error":
        return "Runtime error", "danger"
    state = str(status.get("source_state") or "idle").lower()
    labels = {
        "idle": "Idle",
        "connecting": "Connecting",
        "connected": "Connected",
        "ended": "Completed",
        "stopped": "Stopped",
        "disconnected": "Disconnected",
    }
    tone = "normal" if state == "connected" else "info"
    return labels.get(state, state.replace("_", " ").title()), tone


def temporal_phase(status: dict) -> str:
    """Describe the latch state without changing temporal semantics."""
    if not status.get("event_active"):
        return "Inactive"
    if int(status.get("negative_run", 0) or 0) > 0:
        return "Releasing"
    return "Active"


def event_date(event: dict) -> date | None:
    """Read the date prefix used by persisted event timestamps."""
    value = str(event.get("timestamp") or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


def filter_history(
    events: list[dict],
    *,
    dates: tuple[date, date] | None = None,
    source_or_location: str = "All",
    notification: str = "All",
) -> list[dict]:
    """Apply the three user-facing history filters and keep newest first."""
    filtered: list[dict] = []
    for event in events:
        when = event_date(event)
        if dates and (when is None or not dates[0] <= when <= dates[1]):
            continue
        source_label = compact_source_label(event.get("source"))
        location = str(event.get("location") or "Unknown")
        if source_or_location != "All" and source_or_location not in {source_label, location}:
            continue
        if notification != "All" and str(event.get("notification_status") or "not_attempted") != notification:
            continue
        filtered.append(event)
    return sorted(filtered, key=lambda item: (str(item.get("timestamp") or ""), int(item.get("id", 0) or 0)), reverse=True)


def history_display_rows(events: list[dict]) -> list[dict]:
    """Build the intentionally compact table shown in Event History."""
    return [
        {
            "ID": event.get("id"),
            "Time": event.get("timestamp"),
            "Class": event.get("detected_class", "unknown"),
            "Confidence": float(event.get("confidence", 0.0)),
            "Location": event.get("location") or "Unknown",
            "Source": compact_source_label(event.get("source")),
            "Telegram": notification_label(event),
        }
        for event in events
    ]
