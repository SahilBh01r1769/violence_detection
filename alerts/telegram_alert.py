"""Focused Telegram photo submission without an additional SDK."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


@dataclass(frozen=True)
class TelegramSubmission:
    accepted: bool
    error: Optional[str] = None


def send_telegram_alert(
    *,
    bot_token: str,
    chat_id: str,
    detected_class: str,
    confidence: float,
    screenshot_path: Optional[Path],
    location: str,
    timeout_seconds: float = 15.0,
) -> TelegramSubmission:
    """Submit an event screenshot and caption to Telegram's Bot API.

    ``accepted`` means the API returned an affirmative response. It does not
    establish that a person received or read the message.
    """
    if not bot_token or not chat_id:
        return TelegramSubmission(False, "Telegram credentials are not configured")
    if screenshot_path is None or not Path(screenshot_path).is_file():
        return TelegramSubmission(False, "event screenshot is unavailable")

    caption = (
        f"Violence event\n"
        f"Class: {detected_class}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Source: {location}"
    )
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    try:
        with Path(screenshot_path).open("rb") as image:
            response = requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption},
                files={"photo": (Path(screenshot_path).name, image, "image/jpeg")},
                timeout=timeout_seconds,
            )
    except requests.RequestException as exc:
        # Exception text can include the token-bearing request URL.
        return TelegramSubmission(
            False,
            f"Telegram request failed ({type(exc).__name__})",
        )
    except OSError as exc:
        return TelegramSubmission(False, f"Could not read event screenshot: {exc}")

    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.ok and payload.get("ok") is True:
        return TelegramSubmission(True)

    description = payload.get("description")
    error = str(description) if description else f"Telegram API returned HTTP {response.status_code}"
    return TelegramSubmission(False, error)
