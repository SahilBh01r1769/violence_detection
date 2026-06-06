"""
config.py — Centralised configuration loader
Reads from .env and exposes typed settings across all modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
MODEL_PATH      = os.getenv("MODEL_PATH", "models/violence_yolov8.pt")
SCREENSHOT_DIR  = Path(os.getenv("SCREENSHOT_DIR", "screenshots"))
LOG_DIR         = Path(os.getenv("LOG_DIR", "logs"))

SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Detection ──────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = float(os.getenv("CONFIDENCE_THRESHOLD", 0.55))
FRAME_CONSISTENCY     = int(os.getenv("FRAME_CONSISTENCY", 5))
FPS_TARGET            = int(os.getenv("FPS_TARGET", 20))
VIDEO_SOURCE          = os.getenv("VIDEO_SOURCE", "0")

# Convert "0" → integer 0 for webcam
if VIDEO_SOURCE.isdigit():
    VIDEO_SOURCE = int(VIDEO_SOURCE)

# Detection classes (must match your YOLO model's class names)
VIOLENCE_CLASSES = ["Fighting", "Weapon", "Aggression"]
SAFE_CLASS       = "Normal"

# ── Alerts ─────────────────────────────────────────────────────────────────────
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", 30))
MAX_SCREENSHOTS        = int(os.getenv("MAX_SCREENSHOTS", 500))

# ── Email ──────────────────────────────────────────────────────────────────────
SMTP_HOST      = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", 587))
SMTP_USER      = os.getenv("SMTP_USER", "")
SMTP_PASSWORD  = os.getenv("SMTP_PASSWORD", "")
ALERT_RECIPIENTS = [
    r.strip()
    for r in os.getenv("ALERT_RECIPIENTS", "").split(",")
    if r.strip()
]

# ── Twilio / WhatsApp ──────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID     = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN      = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM   = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TWILIO_WHATSAPP_TO     = os.getenv("TWILIO_WHATSAPP_TO", "")

# ── Server ─────────────────────────────────────────────────────────────────────
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 8501))
API_PORT       = int(os.getenv("API_PORT", 8000))
