"""
api/server.py — FastAPI Backend

Exposes REST endpoints consumed by the Streamlit dashboard and
any external integrations.

Endpoints
---------
GET  /health                     System health check
GET  /status                     Pipeline runtime stats
GET  /alerts                     Paginated alert history
GET  /alerts/{id}/screenshot     Serve a screenshot image
POST /pipeline/start             Start detection
POST /pipeline/stop              Stop detection
POST /pipeline/config            Update confidence / cooldown at runtime
GET  /stream/frame               Latest annotated JPEG frame (MJPEG-style)
"""

from __future__ import annotations

import io
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from config import CONFIDENCE_THRESHOLD, ALERT_COOLDOWN_SECONDS, LOG_DIR
from core.pipeline import DetectionPipeline

logger = logging.getLogger(__name__)
app    = FastAPI(title="Violence Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Pipeline State ────────────────────────────────────────────────────

_pipeline:        Optional[DetectionPipeline] = None
_pipeline_thread: Optional[threading.Thread]  = None
_latest_frame:    Optional[np.ndarray]        = None
_frame_lock       = threading.Lock()


def _frame_callback(annotated_frame, result):
    """Store the latest annotated frame for /stream/frame endpoint."""
    global _latest_frame
    with _frame_lock:
        _latest_frame = annotated_frame


# ── Health / Status ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/status")
def status():
    if _pipeline is None:
        return {"running": False}
    return {
        "running":          _pipeline._running,
        "frames_processed": _pipeline.frames_processed,
        "alerts_fired":     _pipeline.alerts_fired,
        "uptime_seconds":   round(_pipeline.uptime, 1),
        "fps":              round(_pipeline.fps, 2),
        "alert_history_count": len(_pipeline.alert_manager.history),
        "cooldown_remaining":  round(_pipeline.alert_manager.seconds_until_next_alert, 1),
    }


# ── Alerts ───────────────────────────────────────────────────────────────────

@app.get("/alerts")
def get_alerts(
    page:     int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
):
    if _pipeline is None:
        return {"alerts": [], "total": 0}

    history = _pipeline.alert_manager.history
    total   = len(history)
    start   = (page - 1) * per_page
    end     = start + per_page
    page_items = history[start:end]

    return {
        "alerts":   [vars(a) for a in page_items],
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    (total + per_page - 1) // per_page,
    }


@app.get("/alerts/{alert_id}/screenshot")
def get_screenshot(alert_id: int):
    if _pipeline is None:
        raise HTTPException(404, "Pipeline not running")

    for record in _pipeline.alert_manager.history:
        if record.id == alert_id and record.screenshot_path:
            p = Path(record.screenshot_path)
            if p.exists():
                return Response(content=p.read_bytes(), media_type="image/jpeg")
            raise HTTPException(404, "Screenshot file not found on disk")

    raise HTTPException(404, f"Alert {alert_id} not found")


# ── Pipeline Control ─────────────────────────────────────────────────────────

class PipelineStartRequest(BaseModel):
    source:   str = "0"
    location: str = "Camera-01"


@app.post("/pipeline/start")
def start_pipeline(req: PipelineStartRequest):
    global _pipeline, _pipeline_thread

    if _pipeline and _pipeline._running:
        return JSONResponse({"message": "Pipeline already running"}, status_code=200)

    source = int(req.source) if req.source.isdigit() else req.source

    _pipeline = DetectionPipeline(
        on_frame=_frame_callback,
        location=req.location,
        show_window=False,
    )

    _pipeline_thread = threading.Thread(
        target=_pipeline.run,
        kwargs={"source": source},
        daemon=True,
    )
    _pipeline_thread.start()
    logger.info("Pipeline started on source: %s", source)
    return {"message": "Pipeline started", "source": req.source}


@app.post("/pipeline/stop")
def stop_pipeline():
    if _pipeline is None or not _pipeline._running:
        return {"message": "Pipeline is not running"}
    _pipeline.stop()
    return {"message": "Stop signal sent"}


class PipelineConfigRequest(BaseModel):
    confidence:       Optional[float] = None
    cooldown_seconds: Optional[int]   = None


@app.post("/pipeline/config")
def update_config(req: PipelineConfigRequest):
    if _pipeline is None:
        raise HTTPException(400, "Pipeline not running")

    if req.confidence is not None:
        _pipeline.detector.confidence = req.confidence

    if req.cooldown_seconds is not None:
        _pipeline.alert_manager.cooldown = req.cooldown_seconds

    return {"message": "Config updated", "applied": req.dict(exclude_none=True)}


# ── Live Frame ───────────────────────────────────────────────────────────────

@app.get("/stream/frame")
def latest_frame():
    """Return the latest annotated JPEG frame."""
    with _frame_lock:
        frame = _latest_frame

    if frame is None:
        # Return a black placeholder
        import cv2
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder, "No stream available",
            (140, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2,
        )
        frame = placeholder

    import cv2
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
