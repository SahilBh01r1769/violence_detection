"""
alerts/email_alert.py — Email Alert via SMTP

Sends an HTML email with:
  - Alert summary (class, confidence, timestamp, location)
  - Inline screenshot attachment
"""

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

from config import (
    ALERT_RECIPIENTS,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USER,
)

logger = logging.getLogger(__name__)


def send_email_alert(
    detected_class: str,
    confidence: float,
    screenshot_path: Optional[Path] = None,
    location: str = "Camera-01",
    recipients: Optional[List[str]] = None,
) -> bool:
    """
    Send an alert email with an optional screenshot attachment.

    Returns True on success, False on failure.
    """
    recipients = recipients or ALERT_RECIPIENTS
    if not recipients:
        logger.warning("No email recipients configured. Skipping email alert.")
        return False

    if not SMTP_USER or not SMTP_PASSWORD:
        logger.error("SMTP credentials not set. Configure SMTP_USER and SMTP_PASSWORD in .env")
        return False

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    subject   = f"🚨 VIOLENCE ALERT — {detected_class} detected at {location}"

    # ── HTML Body ────────────────────────────────────────────────────────────
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body      {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f0f; color: #e0e0e0; margin: 0; padding: 0; }}
        .wrapper  {{ max-width: 600px; margin: 30px auto; background: #1a1a1a; border-radius: 12px; overflow: hidden; border: 1px solid #333; }}
        .header   {{ background: linear-gradient(135deg, #c0392b, #922b21); padding: 28px 32px; }}
        .header h1{{ margin: 0; font-size: 22px; color: #fff; letter-spacing: 1px; }}
        .header p {{ margin: 6px 0 0; font-size: 13px; color: rgba(255,255,255,.75); }}
        .body     {{ padding: 28px 32px; }}
        .card     {{ background: #252525; border-radius: 8px; padding: 20px; margin-bottom: 20px; border-left: 4px solid #e74c3c; }}
        .row      {{ display: flex; justify-content: space-between; margin-bottom: 10px; }}
        .label    {{ font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: .5px; }}
        .value    {{ font-size: 14px; font-weight: 600; color: #fff; }}
        .badge    {{ display: inline-block; background: #e74c3c; color: #fff; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 700; }}
        .screenshot {{ margin-top: 20px; }}
        .screenshot img {{ width: 100%; border-radius: 8px; border: 1px solid #333; }}
        .footer   {{ padding: 16px 32px; background: #111; font-size: 11px; color: #555; text-align: center; }}
      </style>
    </head>
    <body>
      <div class="wrapper">
        <div class="header">
          <h1>⚠ Violence Detection Alert</h1>
          <p>Automated alert from the AI Surveillance System</p>
        </div>
        <div class="body">
          <div class="card">
            <div class="row">
              <div>
                <div class="label">Detected Class</div>
                <div class="value"><span class="badge">{detected_class}</span></div>
              </div>
              <div>
                <div class="label">Confidence</div>
                <div class="value">{confidence:.1%}</div>
              </div>
            </div>
            <div class="row">
              <div>
                <div class="label">Location</div>
                <div class="value">{location}</div>
              </div>
              <div>
                <div class="label">Timestamp</div>
                <div class="value">{timestamp}</div>
              </div>
            </div>
          </div>

          {"<div class='screenshot'><p style='color:#888;font-size:13px;'>Screenshot at time of detection:</p><img src='cid:screenshot'></div>" if screenshot_path else ""}

          <p style="font-size:13px;color:#666;margin-top:24px;">
            Immediate review is recommended. Log into the dashboard for full alert history and live feed.
          </p>
        </div>
        <div class="footer">
          Violence Detection System &nbsp;|&nbsp; {timestamp} &nbsp;|&nbsp; Automated Alert — Do Not Reply
        </div>
      </div>
    </body>
    </html>
    """

    # ── Build Message ────────────────────────────────────────────────────────
    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"]    = f"Violence Detection System <{SMTP_USER}>"
    msg["To"]      = ", ".join(recipients)

    alternative = MIMEMultipart("alternative")
    alternative.attach(MIMEText(f"ALERT: {detected_class} detected at {location} ({timestamp})", "plain"))
    alternative.attach(MIMEText(html, "html"))
    msg.attach(alternative)

    # Attach screenshot inline
    if screenshot_path and Path(screenshot_path).exists():
        with open(screenshot_path, "rb") as f:
            img = MIMEImage(f.read(), _subtype="jpeg")
            img.add_header("Content-ID", "<screenshot>")
            img.add_header("Content-Disposition", "inline", filename=Path(screenshot_path).name)
            msg.attach(img)

    # ── Send ─────────────────────────────────────────────────────────────────
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, recipients, msg.as_string())
        logger.info("Email alert sent to: %s", recipients)
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed. Check SMTP_USER / SMTP_PASSWORD.")
    except smtplib.SMTPException as exc:
        logger.error("SMTP error: %s", exc)
    except Exception as exc:
        logger.error("Unexpected error sending email: %s", exc)

    return False
