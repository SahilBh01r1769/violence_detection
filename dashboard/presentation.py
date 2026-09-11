"""Small presentation helpers for dashboard event records."""

from __future__ import annotations

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
