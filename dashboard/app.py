"""Thin Streamlit view for pipeline control, diagnostics, and event history."""

from __future__ import annotations

import base64
import csv
import io
import os
import sys
import time
from pathlib import Path

import requests
import streamlit as st

# Keep the repository's config.py ahead of an unrelated installed config package
# when the dashboard is started through the Streamlit console script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.path.remove(str(PROJECT_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ALERT_COOLDOWN_SECONDS,
    CONFIDENCE_THRESHOLD,
    ENABLE_TELEGRAM_ALERTS,
    FRAME_CONSISTENCY,
    NEGATIVE_RELEASE_FRAMES,
    ROLLING_WINDOW_SIZE,
    TEMPORAL_STRATEGY,
)
from dashboard.video_input import persist_uploaded_video

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
st.set_page_config(page_title="Temporal Violence Events", layout="wide")

st.markdown(
    """
<style>
:root {
  --page: #f1eee8;
  --panel: #e7e1d7;
  --panel-2: #ddd6cb;
  --ink: #29251f;
  --muted: #6b645b;
  --line: #bdb4a8;
  --accent: #82503a;
  --slate: #50616d;
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--page);
  color: var(--ink);
  font-family: Arial, Helvetica, sans-serif;
}
[data-testid="stSidebar"] {
  background: #e8e2d8;
  border-right: 1px solid var(--line);
}
h1, h2, h3 { color: var(--ink) !important; letter-spacing: -0.02em; }
[data-testid="stAppViewContainer"] .block-container {
  max-width: 1180px;
  padding-top: 2rem;
}
.hero {
  background: linear-gradient(135deg, #2f3e46 0%, #50616d 62%, #82503a 100%);
  color: #fff;
  padding: 22px 26px;
  border-radius: 10px;
  margin-bottom: 20px;
}
.hero h1 { color: #fff !important; margin: 0 0 6px; font-size: 2rem; }
.hero p { margin: 0; max-width: 780px; color: #eee9e2; }
.section-note { color: var(--muted); margin-top: -8px; margin-bottom: 14px; }
.event-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-left: 5px solid var(--accent);
  border-radius: 7px;
  padding: 14px 16px;
  margin-bottom: 10px;
}
.event-card strong { font-size: 1.05rem; }
[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 3px;
  padding: 12px;
  box-shadow: none;
}
.stButton > button,
.stDownloadButton > button,
[data-baseweb="select"] > div,
[data-testid="stTextInputRootElement"],
[data-testid="stNumberInputContainer"] {
  border-radius: 3px !important;
  box-shadow: none !important;
}
.stButton > button { transition: none !important; }
.stButton > button:hover { transform: none !important; }
[data-testid="stAlert"] { border-radius: 3px; }
.skeleton {
  height: 84px;
  border: 1px solid var(--line);
  background: var(--panel-2);
  border-radius: 3px;
  margin: 8px 0 14px;
  animation: skeletonPulse 1.05s ease-in-out infinite;
}
.skeleton.tall { height: 220px; }
@keyframes skeletonPulse { 0%,100% { opacity: .45; } 50% { opacity: .78; } }
</style>
""",
    unsafe_allow_html=True,
)

def loading_placeholder(tall: bool = False):
    slot = st.empty()
    slot.markdown(f'<div class="skeleton{" tall" if tall else ""}"></div>', unsafe_allow_html=True)
    return slot


def api_get(path: str, default=None):
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=3)
        response.raise_for_status()
        return response.json()
    except Exception:
        return default


def api_post(path: str, payload: dict | None = None):
    try:
        response = requests.post(f"{API_BASE}{path}", json=payload or {}, timeout=30)
        data = response.json() if response.content else {}
        return data if response.ok else {"error": data.get("detail") or data.get("message") or response.text}
    except Exception as exc:
        return {"error": str(exc)}


def live_frame_b64() -> str | None:
    try:
        response = requests.get(f"{API_BASE}/stream/frame", timeout=3)
        response.raise_for_status()
        return base64.b64encode(response.content).decode()
    except Exception:
        return None


def load_event_history() -> list[dict]:
    alerts = []
    page = 1
    while True:
        data = api_get(f"/alerts?page={page}&per_page=100", {"alerts": [], "pages": 0}) or {"alerts": [], "pages": 0}
        alerts.extend(data.get("alerts", []))
        pages = int(data.get("pages", 0) or 0)
        if page >= pages:
            break
        page += 1
    return alerts


def events_as_csv(events: list[dict]) -> str:
    if not events:
        return ""
    fieldnames = [key for key in events[0] if key != "screenshot_path"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(events)
    return output.getvalue()


def event_screenshot(event_id: int) -> bytes | None:
    try:
        response = requests.get(
            f"{API_BASE}/alerts/{event_id}/screenshot",
            timeout=3,
        )
        response.raise_for_status()
        return response.content
    except Exception:
        return None


def setting(name: str, default):
    return st.session_state.get(name, default)


def start_payload(source: str) -> dict:
    return {
        "source": source,
        "location": setting("location", "Camera-01"),
        "confidence": float(setting("confidence", CONFIDENCE_THRESHOLD)),
        "frame_consistency": int(setting("frame_consistency", FRAME_CONSISTENCY)),
        "negative_release_frames": int(
            setting("negative_release_frames", NEGATIVE_RELEASE_FRAMES)
        ),
        "temporal_strategy": setting("temporal_strategy", TEMPORAL_STRATEGY),
        "rolling_window_size": int(
            setting("rolling_window_size", ROLLING_WINDOW_SIZE)
        ),
        "cooldown_seconds": int(setting("cooldown", ALERT_COOLDOWN_SECONDS)),
        "enable_telegram": bool(
            setting("enable_telegram", ENABLE_TELEGRAM_ALERTS)
        ),
    }


with st.sidebar:
    st.title("Temporal Violence Events")
    page = st.radio("Navigation", ["Pipeline", "Event History", "Settings"])
    st.subheader("Input")
    source_kind = st.selectbox(
        "Source type",
        ["Webcam", "Upload video", "Local video path", "RTSP"],
    )
    source_ready = True
    if source_kind == "Upload video":
        upload = st.file_uploader(
            "Video file",
            type=["avi", "mkv", "mov", "mp4", "webm"],
        )
        if upload is None:
            selected_source = ""
            source_ready = False
            st.caption("Choose a video before starting.")
        else:
            try:
                selected_source = str(
                    persist_uploaded_video(upload.name, upload.getvalue())
                )
                st.caption(f"Ready: {Path(selected_source).name}")
            except (OSError, ValueError) as exc:
                selected_source = ""
                source_ready = False
                st.error(str(exc))
    elif source_kind == "Local video path":
        selected_source = st.text_input(
            "Local path",
            value="sample_videos/nonviolence.mp4",
            key="local_video_path",
        )
        source_ready = bool(selected_source.strip())
    elif source_kind == "Webcam":
        selected_source = str(
            st.number_input("Camera index", min_value=0, value=0, step=1)
        )
    else:
        selected_source = st.text_input("RTSP URL", key="rtsp_url")
        source_ready = bool(selected_source.strip())

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "Start",
            type="primary",
            use_container_width=True,
            disabled=not source_ready,
        ):
            result = api_post(
                "/pipeline/start",
                start_payload(str(selected_source)),
            )
            if result.get("error"):
                st.error(result["error"])
            else:
                st.success(result.get("message", "Pipeline started"))
    with c2:
        if st.button("Stop", use_container_width=True):
            result = api_post("/pipeline/stop")
            if result.get("error"):
                st.error(result["error"])
            else:
                st.info(result.get("message", "Pipeline stopped"))
    status = api_get("/status", {}) or {}
    st.caption("ACTIVE" if status.get("running") else "OFFLINE")
    st.caption(f"API: {API_BASE}")

if page == "Pipeline":
    st.markdown(
        """
        <div class="hero">
          <h1>Temporal Violence Events</h1>
          <p>Frame-level model predictions are qualified into persistent events. Local recording is canonical; Telegram is an optional notification path.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status = api_get("/status", {}) or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Source", str(status.get("source_state", "idle")).title())
    m2.metric("Frames", f"{status.get('frames_processed', 0):,}")
    m3.metric("Current events", status.get("events_recorded", 0))
    m4.metric("FPS", f"{status.get('fps', 0):.1f}")

    st.subheader("Current frame")
    st.markdown('<p class="section-note">Bounding boxes show the checkpoint class and confidence. This model knows only violence and non-violence classes.</p>', unsafe_allow_html=True)
    frame_loading = loading_placeholder(tall=True)
    frame_b64 = live_frame_b64()
    frame_loading.empty()
    if frame_b64:
        st.markdown(
            f'<img src="data:image/jpeg;base64,{frame_b64}" style="width:100%;max-width:960px;border:1px solid #bdb4a8">',
            unsafe_allow_html=True,
        )
    else:
        st.info("No frame available. Start the API and pipeline first.")

    runtime_error = status.get("last_error")
    if runtime_error:
        st.error(
            f"{runtime_error.get('stage', 'pipeline').title()} failure: "
            f"{runtime_error.get('message', 'unknown error')}"
        )

    latest_notification = status.get("latest_notification")
    if latest_notification:
        notification_text = (
            f"Telegram: {latest_notification.get('status', 'unknown')}"
        )
        if latest_notification.get("error"):
            notification_text += f" — {latest_notification['error']}"
        st.write(notification_text)
        st.caption(
            "Accepted means Telegram accepted the submission; it does not "
            "establish that a person read it."
        )

    st.subheader("Temporal state")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Event", "Active" if status.get("event_active") else "Idle")
    strategy = status.get("temporal_strategy", setting("temporal_strategy", TEMPORAL_STRATEGY))
    t2.metric(
        "Window positives" if strategy == "rolling_window" else "Positive run",
        f"{status.get('positive_run', 0)} / {status.get('frame_consistency', setting('frame_consistency', FRAME_CONSISTENCY))}",
    )
    t3.metric(
        "Negative run",
        f"{status.get('negative_run', 0)} / {status.get('negative_release_frames', setting('negative_release_frames', NEGATIVE_RELEASE_FRAMES))}",
    )
    t4.metric("Cooldown", f"{status.get('cooldown_remaining', 0):.0f}s")
    qualification_text = (
        f"M positives within W={status.get('rolling_window_size', setting('rolling_window_size', ROLLING_WINDOW_SIZE))} frames start an event; "
        if strategy == "rolling_window"
        else "N consecutive positive frames start an event; "
    )
    st.caption(
        f"Decision confidence ≥ {status.get('confidence', setting('confidence', CONFIDENCE_THRESHOLD)):.2f}. "
        + qualification_text
        + "K negative frames end it."
    )

    st.subheader("Events from this run")
    st.markdown('<p class="section-note">Older records remain available on the Event History page.</p>', unsafe_allow_html=True)
    current = api_get("/alerts/current", {"alerts": []}) or {"alerts": []}
    current_events = current.get("alerts", [])
    if not current_events:
        st.info("No event has been qualified in this run.")
    else:
        for event in current_events[:6]:
            notification = event.get("notification_status", "not_attempted").replace("_", " ").title()
            st.markdown(
                f'<div class="event-card"><strong>Event #{event["id"]} · {event.get("detected_class", "unknown")}</strong><br>'
                f'{float(event.get("confidence", 0)):.0%} confidence · {event.get("timestamp", "unknown time")} · Telegram: {notification}</div>',
                unsafe_allow_html=True,
            )
            screenshot = event_screenshot(int(event["id"]))
            if screenshot:
                st.image(screenshot, width=420)
    if st.checkbox("Refresh frame and status every second", value=False):
        time.sleep(1)
        st.rerun()

elif page == "Event History":
    st.header("Event History")
    loading = loading_placeholder(tall=True)
    events = load_event_history()
    loading.empty()
    if not events:
        st.info("No event records available yet.")
    else:
        classes = ["All"] + sorted(
            {
                event.get("detected_class")
                for event in events
                if event.get("detected_class")
            }
        )
        c1, c2 = st.columns(2)
        class_filter = c1.selectbox("Class", classes)
        min_confidence = c2.slider(
            "Minimum confidence",
            0.0,
            1.0,
            0.0,
            0.05,
        )
        filtered = [
            event
            for event in events
            if float(event.get("confidence", 0.0)) >= min_confidence
            and (
                class_filter == "All"
                or event.get("detected_class") == class_filter
            )
        ]
        displayed = [
            {
                key: value
                for key, value in event.items()
                if key != "screenshot_path"
            }
            for event in filtered
        ]
        st.dataframe(displayed, use_container_width=True, hide_index=True)
        st.download_button(
            "Export CSV",
            events_as_csv(filtered),
            file_name="event_history.csv",
            mime="text/csv",
        )
        st.subheader("Screenshots")
        for event in filtered[:20]:
            screenshot = event_screenshot(int(event["id"]))
            if screenshot:
                st.image(
                    screenshot,
                    caption=(
                        f"Event #{event['id']}: "
                        f"{event.get('detected_class', 'unknown')}"
                    ),
                )

elif page == "Settings":
    st.header("Settings")
    status = api_get("/status", {}) or {}
    with st.form("settings"):
        confidence = st.slider("Confidence threshold", 0.05, 1.0, float(setting("confidence", CONFIDENCE_THRESHOLD)), 0.05)
        frame_consistency = st.number_input("Positive observations required (N or M)", 1, 120, int(setting("frame_consistency", FRAME_CONSISTENCY)))
        temporal_strategy = st.selectbox(
            "Temporal strategy",
            ["consecutive", "rolling_window"],
            index=0 if setting("temporal_strategy", TEMPORAL_STRATEGY) == "consecutive" else 1,
            format_func=lambda value: "Consecutive frames" if value == "consecutive" else "Rolling-window voting",
        )
        rolling_window_size = st.number_input(
            "Rolling window size W",
            int(frame_consistency),
            120,
            int(setting("rolling_window_size", ROLLING_WINDOW_SIZE)),
            disabled=temporal_strategy != "rolling_window",
        )
        negative_release_frames = st.number_input(
            "Consecutive negative frames to end an event",
            1,
            120,
            int(setting("negative_release_frames", NEGATIVE_RELEASE_FRAMES)),
        )
        cooldown = st.number_input("Alert cooldown (seconds)", 0, 86400, int(setting("cooldown", ALERT_COOLDOWN_SECONDS)))
        location = st.text_input("Camera/location label", setting("location", "Camera-01"))
        enable_telegram = st.checkbox(
            "Enable Telegram notifications",
            value=bool(setting("enable_telegram", ENABLE_TELEGRAM_ALERTS)),
        )
        submitted = st.form_submit_button("Save and Apply")
    if submitted:
        st.session_state.update(
            confidence=confidence,
            frame_consistency=int(frame_consistency),
            temporal_strategy=temporal_strategy,
            rolling_window_size=int(rolling_window_size),
            negative_release_frames=int(negative_release_frames),
            cooldown=int(cooldown),
            location=location,
            enable_telegram=enable_telegram,
        )
        if status.get("running"):
            result = api_post(
                "/pipeline/config",
                {
                    "confidence": confidence,
                    "frame_consistency": int(frame_consistency),
                    "temporal_strategy": temporal_strategy,
                    "rolling_window_size": int(rolling_window_size),
                    "negative_release_frames": int(negative_release_frames),
                    "cooldown_seconds": int(cooldown),
                    "enable_telegram": enable_telegram,
                },
            )
            if result.get("error"):
                st.error(result["error"])
            else:
                st.success("Settings applied to the running pipeline.")
        else:
            st.success("Settings saved. They will be used the next time the pipeline starts.")
    st.caption(
        "Choose the input in the sidebar. Location changes take effect on the "
        "next pipeline start."
    )
