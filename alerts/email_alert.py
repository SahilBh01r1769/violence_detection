"""SMTP Email alert sender."""

from __future__ import annotations

import logging
import smtplib
import ssl
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Optional

from config import ALERT_RECIPIENTS, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USER

logger = logging.getLogger(__name__)


def send_email_alert(detected_class: str, confidence: float, screenshot_path: Optional[Path] = None, location: str = "Camera-01", recipients: Optional[List[str]] = None) -> bool:
    recipients = recipients or ALERT_RECIPIENTS
    if not recipients:
        logger.warning("No email recipients configured; skipping email alert")
        return False
    if not SMTP_USER or not SMTP_PASSWORD:
        logger.warning("SMTP credentials are not configured; skipping email alert")
        return False
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = MIMEMultipart("related")
    msg["Subject"] = f"VIOLENCE ALERT — {detected_class} at {location}"
    msg["From"] = f"Violence Detection System <{SMTP_USER}>"
    msg["To"] = ", ".join(recipients)
    body = f"<h2>Violence Detection Alert</h2><p><b>Class:</b> {detected_class}</p><p><b>Confidence:</b> {confidence:.1%}</p><p><b>Location:</b> {location}</p><p><b>Time:</b> {timestamp}</p>"
    if screenshot_path:
        body += "<p><img src='cid:screenshot' style='max-width:100%'></p>"
    alternative = MIMEMultipart("alternative")
    alternative.attach(MIMEText(f"ALERT: {detected_class} at {location} ({confidence:.1%})", "plain"))
    alternative.attach(MIMEText(body, "html"))
    msg.attach(alternative)
    if screenshot_path and Path(screenshot_path).exists():
        with open(screenshot_path, "rb") as handle:
            image = MIMEImage(handle.read(), _subtype="jpeg")
            image.add_header("Content-ID", "<screenshot>")
            image.add_header("Content-Disposition", "inline", filename=Path(screenshot_path).name)
            msg.attach(image)
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, recipients, msg.as_string())
        logger.info("Email alert sent to %s", recipients)
        return True
    except Exception as exc:
        logger.error("Email alert failed: %s", exc)
        return False
