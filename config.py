"""Central configuration for the violence detection application."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent


def _path_from_env(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default)).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def _csv(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _int_csv(name: str, default: str) -> set[int]:
    values: set[int] = set()
    for item in _csv(name, default):
        try:
            values.add(int(item))
        except ValueError as exc:
            raise ValueError(f"{name} must contain comma-separated integers") from exc
    return values


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


UPSTREAM_MODEL_COMMIT = "20f0d05054cff7da2dc78dee3c2de1bd54106a13"
MODEL_PATH = _path_from_env("MODEL_PATH", "models/violence_yolov8n.pt")
MODEL_DOWNLOAD_URL = os.getenv(
    "MODEL_DOWNLOAD_URL",
    f"https://raw.githubusercontent.com/Musawer1214/Fight-Violence-detection-yolov8/{UPSTREAM_MODEL_COMMIT}/Yolo_nano_weights.pt",
)
AUTO_DOWNLOAD_MODEL = _bool("AUTO_DOWNLOAD_MODEL", True)
MODEL_SOURCE_URL = "https://github.com/Musawer1214/Fight-Violence-detection-yolov8"
SCREENSHOT_DIR = _path_from_env("SCREENSHOT_DIR", "screenshots")
LOG_DIR = _path_from_env("LOG_DIR", "logs")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
FRAME_CONSISTENCY = int(os.getenv("FRAME_CONSISTENCY", "5"))
FPS_TARGET = int(os.getenv("FPS_TARGET", "20"))
VIDEO_SOURCE: int | str = os.getenv("VIDEO_SOURCE", "0")
if isinstance(VIDEO_SOURCE, str) and VIDEO_SOURCE.isdigit():
    VIDEO_SOURCE = int(VIDEO_SOURCE)

VIOLENCE_CLASS_IDS = _int_csv("VIOLENCE_CLASS_IDS", "1")
VIOLENCE_CLASSES = _csv("VIOLENCE_CLASSES", "violence,fight,fighting,violence/fight")
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))
MAX_SCREENSHOTS = int(os.getenv("MAX_SCREENSHOTS", "500"))
ENABLE_EMAIL_ALERTS = _bool("ENABLE_EMAIL_ALERTS", True)
ENABLE_WHATSAPP_ALERTS = _bool("ENABLE_WHATSAPP_ALERTS", True)

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_RECIPIENTS = _csv("ALERT_RECIPIENTS", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TWILIO_WHATSAPP_TO = os.getenv("TWILIO_WHATSAPP_TO", "")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE = os.getenv("API_BASE", f"http://localhost:{API_PORT}")
