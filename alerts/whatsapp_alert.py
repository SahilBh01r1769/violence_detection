"""
alerts/whatsapp_alert.py — WhatsApp Alert via Twilio

Sends a formatted WhatsApp message with key detection details.
Optionally attaches a public media URL if the screenshot is hosted.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from config import (
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_WHATSAPP_FROM,
    TWILIO_WHATSAPP_TO,
)

logger = logging.getLogger(__name__)


def send_whatsapp_alert(
    detected_class: str,
    confidence: float,
    screenshot_path: Optional[Path] = None,
    location: str = "Camera-01",
    media_url: Optional[str] = None,
) -> bool:
    """
    Send a WhatsApp alert message via Twilio.

    Parameters
    ----------
    detected_class  : Name of the detected violence class
    confidence      : Model confidence score (0–1)
    screenshot_path : Local path to screenshot (used for logging; Twilio needs a public URL)
    location        : Camera / location label
    media_url       : Public HTTPS URL of the screenshot (optional, for MMS attachment)

    Returns True on success, False on failure.
    """
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logger.error("Twilio credentials not configured. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env")
        return False

    if not TWILIO_WHATSAPP_TO:
        logger.warning("No WhatsApp recipient configured. Set TWILIO_WHATSAPP_TO in .env")
        return False

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    body = (
        f"🚨 *VIOLENCE ALERT*\n\n"
        f"📍 *Location:* {location}\n"
        f"⚠️ *Detected:* {detected_class}\n"
        f"📊 *Confidence:* {confidence:.1%}\n"
        f"🕐 *Time:* {timestamp}\n\n"
        f"Immediate review recommended. Check the monitoring dashboard for details."
    )

    if screenshot_path:
        body += f"\n\n📸 Screenshot: {Path(screenshot_path).name}"

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        msg_kwargs = {
            "from_": TWILIO_WHATSAPP_FROM,
            "to":    TWILIO_WHATSAPP_TO,
            "body":  body,
        }

        # Only attach media if a public URL is provided
        # (Twilio cannot access local filesystem paths)
        if media_url:
            msg_kwargs["media_url"] = [media_url]

        message = client.messages.create(**msg_kwargs)
        logger.info("WhatsApp alert sent. SID: %s", message.sid)
        return True

    except ImportError:
        logger.error("twilio package not installed. Run: pip install twilio")
    except Exception as exc:
        logger.error("WhatsApp alert failed: %s", exc)

    return False
