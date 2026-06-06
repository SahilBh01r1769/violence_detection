"""
utils/logger.py — Centralised Logging Setup

Call setup_logging() once at application entry points.
Writes to both console and a rotating log file.
"""

import logging
import logging.handlers
from pathlib import Path

from config import LOG_DIR


def setup_logging(
    level: int = logging.INFO,
    log_file: str = "violence_detection.log",
) -> None:
    """
    Configure root logger with:
      - StreamHandler (console) — INFO and above
      - RotatingFileHandler   — all levels, 5 MB × 3 backups
    """
    log_path = LOG_DIR / log_file

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)-30s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File (rotating)
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
