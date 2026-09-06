"""FastAPI backend for pipeline control, status, history and live frames."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from alerts.alert_manager import ALERT_LOG_FILE, AlertRecord
from config import (
    ALERT_COOLDOWN_SECONDS,
    CONFIDENCE_THRESHOLD,
    ENABLE_EMAIL_ALERTS,
    ENABLE_WHATSAPP_ALERTS,
    FRAME_CONSISTENCY,
    NEGATIVE_RELEASE_FRAMES,
    VIDEO_SOURCE,
)
from core.pipeline import DetectionPipeline

logger = logging.getLogger(__name__)
app = FastAPI(title="Violence Detection API", version="1.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_pipeline: Optional[DetectionPipeline] = None
_pipeline_thread: Optional[threading.Thread] = None
_latest_frame: Optional[np.ndarray] = None
_frame_lock = threading.Lock()
_pipeline_lock = threading.Lock()


def _history_records():
    if _pipeline is not None:
        return _pipeline.alert_manager.history
    if not ALERT_LOG_FILE.exists():
        return []
    try:
        data = json.loads(ALERT_LOG_FILE.read_text(encoding="utf-8"))
        return list(reversed([AlertRecord(**record) for record in data]))
    except Exception as exc:
        logger.warning("Could not read persisted alert history: %s", exc)
        return []


def _clear_latest_frame():
    global _latest_frame
    with _frame_lock:
        _latest_frame = None


def _frame_callback(annotated_frame, _result):
    global _latest_frame
    with _frame_lock:
        _latest_frame = annotated_frame.copy()


def _latest_delivery(history):
    if not history:
        return None
    alert = history[0]
    return {
        "id": alert.id,
        "status": alert.status,
        "completed_at": alert.completed_at,
        "error": alert.error,
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/status")
def status():
    history = _history_records()
    if _pipeline is None:
        return {
            "running": False,
            "source_state": "idle",
            "last_error": None,
            "latest_alert_delivery": _latest_delivery(history),
            "alert_history_count": len(history),
        }
    return {
        "running": _pipeline._running,
        "source_state": _pipeline.source_state,
        "last_error": _pipeline.last_error.as_dict() if _pipeline.last_error else None,
        "latest_alert_delivery": _latest_delivery(history),
        "frames_processed": _pipeline.frames_processed,
        "alerts_fired": _pipeline.alerts_fired,
        "uptime_seconds": round(_pipeline.uptime, 1),
        "fps": round(_pipeline.fps, 2),
        "alert_history_count": len(history),
        "cooldown_remaining": round(_pipeline.alert_manager.seconds_until_next_alert, 1),
        "cooldown_seconds": _pipeline.alert_manager.cooldown,
        "confidence": _pipeline.detector.confidence,
        "frame_consistency": _pipeline.detector.frame_consistency,
        "negative_release_frames": _pipeline.detector.negative_release_frames,
        "event_active": _pipeline.detector.event_active,
        "email_enabled": _pipeline.alert_manager.enable_email,
        "whatsapp_enabled": _pipeline.alert_manager.enable_whatsapp,
    }


@app.get("/alerts")
def get_alerts(page: int = Query(1, ge=1), per_page: int = Query(20, ge=1, le=100)):
    history = _history_records()
    total = len(history)
    start = (page - 1) * per_page
    items = history[start:start + per_page]
    return {"alerts": [vars(alert) for alert in items], "total": total, "page": page, "per_page": per_page, "pages": (total + per_page - 1) // per_page}


@app.get("/alerts/{alert_id}/screenshot")
def get_screenshot(alert_id: int):
    for record in _history_records():
        if record.id == alert_id and record.screenshot_path:
            path = Path(record.screenshot_path)
            if path.exists():
                return Response(content=path.read_bytes(), media_type="image/jpeg")
            raise HTTPException(404, "Screenshot file not found on disk")
    raise HTTPException(404, f"Alert {alert_id} not found")


class PipelineStartRequest(BaseModel):
    source: str = str(VIDEO_SOURCE)
    location: str = "Camera-01"
    confidence: float = Field(CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
    frame_consistency: int = Field(FRAME_CONSISTENCY, ge=1, le=120)
    negative_release_frames: int = Field(NEGATIVE_RELEASE_FRAMES, ge=1, le=120)
    cooldown_seconds: int = Field(ALERT_COOLDOWN_SECONDS, ge=0, le=86400)
    enable_email: bool = ENABLE_EMAIL_ALERTS
    enable_whatsapp: bool = ENABLE_WHATSAPP_ALERTS


@app.post("/pipeline/start")
def start_pipeline(req: PipelineStartRequest):
    global _pipeline, _pipeline_thread
    with _pipeline_lock:
        if _pipeline and _pipeline._running:
            return JSONResponse({"message": "Pipeline already running"}, status_code=200)
        if _pipeline_thread and _pipeline_thread.is_alive():
            _pipeline_thread.join(timeout=5)
            if _pipeline_thread.is_alive():
                raise HTTPException(409, "Previous pipeline is still shutting down; try again shortly")
        _clear_latest_frame()
        source = int(req.source) if req.source.isdigit() else req.source
        try:
            _pipeline = DetectionPipeline(
                on_frame=_frame_callback,
                location=req.location,
                show_window=False,
                confidence=req.confidence,
                frame_consistency=req.frame_consistency,
                negative_release_frames=req.negative_release_frames,
                cooldown_seconds=req.cooldown_seconds,
                enable_email=req.enable_email,
                enable_whatsapp=req.enable_whatsapp,
            )
        except Exception as exc:
            logger.exception("Could not initialise pipeline")
            raise HTTPException(500, f"Could not initialise pipeline: {exc}") from exc
        _pipeline_thread = threading.Thread(target=_pipeline.run, kwargs={"source": source}, daemon=True, name="violence-detection-pipeline")
        _pipeline_thread.start()
        return {"message": "Pipeline started", "source": req.source}


@app.post("/pipeline/stop")
def stop_pipeline():
    if _pipeline is None or not _pipeline._running:
        _clear_latest_frame()
        return {"message": "Pipeline is not running"}
    _pipeline.stop()
    if _pipeline_thread and _pipeline_thread.is_alive():
        _pipeline_thread.join(timeout=5)
    _clear_latest_frame()
    return {"message": "Pipeline stopped" if not (_pipeline_thread and _pipeline_thread.is_alive()) else "Stop signal sent"}


class PipelineConfigRequest(BaseModel):
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    frame_consistency: Optional[int] = Field(None, ge=1, le=120)
    negative_release_frames: Optional[int] = Field(None, ge=1, le=120)
    cooldown_seconds: Optional[int] = Field(None, ge=0, le=86400)
    enable_email: Optional[bool] = None
    enable_whatsapp: Optional[bool] = None


@app.post("/pipeline/config")
def update_config(req: PipelineConfigRequest):
    if _pipeline is None:
        raise HTTPException(400, "Pipeline has not been started")
    if req.confidence is not None:
        _pipeline.detector.confidence = req.confidence
    if req.frame_consistency is not None:
        _pipeline.detector.set_frame_consistency(req.frame_consistency)
    if req.negative_release_frames is not None:
        _pipeline.detector.set_negative_release_frames(req.negative_release_frames)
    if req.cooldown_seconds is not None:
        _pipeline.alert_manager.cooldown = req.cooldown_seconds
    if req.enable_email is not None:
        _pipeline.alert_manager.enable_email = req.enable_email
    if req.enable_whatsapp is not None:
        _pipeline.alert_manager.enable_whatsapp = req.enable_whatsapp
    return {"message": "Config updated", "applied": req.model_dump(exclude_none=True)}


@app.get("/stream/frame")
def latest_frame():
    import cv2
    with _frame_lock:
        frame = None if _latest_frame is None else _latest_frame.copy()
    if frame is None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "No stream available", (140, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(500, "Could not encode frame")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
