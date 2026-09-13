"""Compact Streamlit client for runtime control and persisted event review."""

from __future__ import annotations

import base64
import csv
import html
import io
import os
import sys
import time
from datetime import date
from pathlib import Path

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    ALERT_COOLDOWN_SECONDS, CONFIDENCE_THRESHOLD, ENABLE_TELEGRAM_ALERTS,
    FRAME_CONSISTENCY, NEGATIVE_RELEASE_FRAMES, ROLLING_WINDOW_SIZE,
    TEMPORAL_STRATEGY,
)
from dashboard.presentation import (  # noqa: E402
    compact_source_label, event_date, filter_history, format_duration,
    notification_label, runtime_state_label, temporal_phase,
)
from dashboard.video_input import persist_uploaded_video  # noqa: E402

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
st.set_page_config(page_title="Temporal Violence Events", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
<style>
:root{--page:#151A1D;--panel:#1B2125;--raised:#20272B;--border:#3B474D;--text:#E7ECEF;--muted:#A6B0B5;--amber:#D79A4A;--red:#C95B5B;--green:#6F9B88;--blue:#7897AE}
html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:var(--page);color:var(--text);font-family:Inter,Arial,sans-serif}
[data-testid="stAppViewContainer"] .block-container{max-width:1320px;padding:1.2rem 2rem 3rem}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none} h1,h2,h3,label,p,span,div{color:var(--text)}
h1{font-size:1.55rem!important;margin:0!important;letter-spacing:-.02em} h2{font-size:1.2rem!important;margin-top:1.5rem!important} h3{font-size:1rem!important}
.kicker{color:var(--amber);font-size:.72rem;letter-spacing:.11em;text-transform:uppercase}.note{color:var(--muted);font-size:.82rem;margin:-.55rem 0 .75rem}
hr{border-color:var(--border)!important}[data-testid="stHorizontalBlock"]{align-items:stretch}
[data-testid="stMetric"]{background:transparent;border:0;padding:.15rem .7rem}[data-testid="stMetricLabel"] p{color:var(--muted)!important;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em}[data-testid="stMetricValue"]{font-size:1.05rem}
.runtime{margin:.9rem 0 1rem}
.frame{background:#090C0E;border:1px solid var(--border);border-radius:5px;padding:8px;min-height:260px}.frame img{display:block;width:100%;max-height:610px;object-fit:contain}
.empty{min-height:330px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:.86rem}
.evidence{background:var(--raised);border:1px solid var(--border);border-radius:5px;padding:.65rem .8rem;margin-top:.7rem}
.event{background:var(--panel);border:1px solid var(--border);border-left:3px solid var(--red);border-radius:5px;padding:.75rem .9rem;margin-bottom:.55rem}.meta{color:var(--muted);font-size:.82rem;line-height:1.55}
.chip{display:inline-block;border:1px solid var(--border);border-radius:3px;padding:.17rem .45rem;font-size:.72rem;font-weight:650}.chip.normal{color:var(--green);border-color:var(--green)}.chip.info{color:var(--blue);border-color:var(--blue)}.chip.danger{color:var(--red);border-color:var(--red)}
[data-testid="stForm"],[data-testid="stExpander"],[data-testid="stVerticalBlockBorderWrapper"]{background:var(--panel);border-color:var(--border)!important;border-radius:6px;box-shadow:none}[data-testid="stForm"]{padding:.9rem}
[data-testid="stAlert"]{border-radius:4px;border:1px solid var(--border);background:var(--raised)}
.stButton>button,.stDownloadButton>button,[data-baseweb="select"]>div,[data-testid="stTextInputRootElement"],[data-testid="stNumberInputContainer"],[data-testid="stFileUploaderDropzone"]{background:var(--raised)!important;color:var(--text)!important;border-radius:4px!important;box-shadow:none!important;border-color:var(--border)!important}
[data-baseweb="popover"],[data-baseweb="popover"]>div,[role="listbox"],[role="option"],[data-baseweb="menu"]{background:#252D31!important;color:var(--text)!important}
[role="option"]:hover,[aria-selected="true"]{background:#344047!important}[role="option"] span,[role="option"] div{color:var(--text)!important}
input,textarea,[data-baseweb="select"] span,[data-baseweb="select"] svg{color:var(--text)!important;fill:var(--text)!important}
.stButton>button[kind="primary"]{background:var(--amber);color:#101417;border-color:var(--amber);font-weight:700}.stButton>button:hover{transform:none!important;border-color:var(--amber)!important}
[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:4px}@media(max-width:760px){[data-testid="stAppViewContainer"] .block-container{padding:1rem}}
</style>
""", unsafe_allow_html=True,
)


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
    alerts, page_number = [], 1
    while True:
        data = api_get(f"/alerts?page={page_number}&per_page=100", {"alerts": [], "pages": 0}) or {"alerts": [], "pages": 0}
        alerts.extend(data.get("alerts", []))
        if page_number >= int(data.get("pages", 0) or 0):
            return alerts
        page_number += 1


def events_as_csv(events: list[dict]) -> str:
    if not events:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=[key for key in events[0] if key != "screenshot_path"], extrasaction="ignore")
    writer.writeheader(); writer.writerows(events)
    return output.getvalue()


def event_screenshot(event_id: int) -> bytes | None:
    try:
        response = requests.get(f"{API_BASE}/alerts/{event_id}/screenshot", timeout=3)
        response.raise_for_status()
        return response.content
    except Exception:
        return None


def setting(name: str, default):
    return st.session_state.get(name, default)


def start_payload(source: str) -> dict:
    return {
        "source": source, "location": setting("location", "Camera-01"),
        "confidence": float(setting("confidence", CONFIDENCE_THRESHOLD)),
        "frame_consistency": int(setting("frame_consistency", FRAME_CONSISTENCY)),
        "negative_release_frames": int(setting("negative_release_frames", NEGATIVE_RELEASE_FRAMES)),
        "temporal_strategy": setting("temporal_strategy", TEMPORAL_STRATEGY),
        "rolling_window_size": int(setting("rolling_window_size", ROLLING_WINDOW_SIZE)),
        "cooldown_seconds": int(setting("cooldown", ALERT_COOLDOWN_SECONDS)),
        "enable_telegram": bool(setting("enable_telegram", ENABLE_TELEGRAM_ALERTS)),
    }


def source_picker(disabled: bool) -> tuple[str, bool]:
    kind = st.selectbox("Source type", ["Webcam", "Upload video", "Local video path", "RTSP"], disabled=disabled)
    if kind == "Upload video":
        upload = st.file_uploader("Video file", type=["avi", "mkv", "mov", "mp4", "webm"], disabled=disabled)
        if upload is None:
            st.caption("Choose a video file to continue."); return "", False
        try:
            path = str(persist_uploaded_video(upload.name, upload.getvalue()))
            st.caption(f"Ready · {Path(path).name}"); return path, True
        except (OSError, ValueError) as exc:
            st.error(str(exc)); return "", False
    if kind == "Local video path":
        source = st.text_input("Local video path", value="sample_videos/nonviolence.mp4", key="local_video_path", disabled=disabled)
        return source, bool(source.strip())
    if kind == "Webcam":
        return str(st.number_input("Camera index", min_value=0, value=0, step=1, disabled=disabled)), True
    source = st.text_input("RTSP URL", key="rtsp_url", disabled=disabled, type="password")
    return source, bool(source.strip())


def render_header() -> str:
    title, nav = st.columns([1.05, 1.35], vertical_alignment="center")
    with title:
        st.markdown('<div class="kicker">Applied ML systems experiment</div>', unsafe_allow_html=True)
        st.title("Temporal Violence Events")
    with nav:
        return st.radio("View", ["Monitor", "Event history", "Settings"], horizontal=True, label_visibility="collapsed")


def render_runtime(status: dict) -> None:
    label, tone = runtime_state_label(status)
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([1.05, 1.55, 1, 1, 1])
        c1.markdown(f'<span class="chip {tone}">{label}</span>', unsafe_allow_html=True)
        c2.metric("Source", compact_source_label(st.session_state.get("run_source", "Not selected")))
        c3.metric("Frames", f"{int(status.get('frames_processed', 0) or 0):,}")
        c4.metric("Events", int(status.get("events_recorded", 0) or 0))
        c5.metric("Elapsed", format_duration(status.get("uptime_seconds", 0)))


def render_event(event: dict, divider: bool = False) -> None:
    event_class = html.escape(str(event.get("detected_class", "unknown")).replace("_", " ").title())
    event_time = html.escape(str(event.get("timestamp", "Unknown")))
    event_location = html.escape(str(event.get("location") or "Unknown"))
    event_source = html.escape(compact_source_label(event.get("source")))
    image_col, detail_col = st.columns([1.65, 1])
    with image_col:
        screenshot = event_screenshot(int(event["id"]))
        if screenshot: st.image(screenshot, use_container_width=True)
        else: st.caption("Screenshot is no longer available in bounded local storage.")
    with detail_col:
        with st.container(border=True):
            st.markdown(f"**Event #{event.get('id')} · {event_class}**")
            st.markdown('<div class="meta">' +
                f"Confidence&nbsp;&nbsp; {float(event.get('confidence', 0)):.0%}<br>" +
                f"Time&nbsp;&nbsp; {event_time}<br>" +
                f"Location&nbsp;&nbsp; {event_location}<br>" +
                f"Source&nbsp;&nbsp; {event_source}</div>", unsafe_allow_html=True)
            st.caption("Notification outcome"); st.write(notification_label(event))
            if event.get("notification_error"): st.error(str(event["notification_error"]))
    if divider: st.divider()


status = api_get("/status", {}) or {}
page = render_header()

if page == "Monitor":
    render_runtime(status)
    running = bool(status.get("running"))
    controls, video = st.columns([.30, .70], gap="large")
    with controls:
        st.subheader("Source")
        if running: st.caption("The current run owns these controls until it stops.")
        selected_source, ready = source_picker(running)
        location = st.text_input("Location label", value=setting("location", "Camera-01"), disabled=running, help="Stored with every qualified event.")
        if not running:
            st.session_state.location = location
            if st.button("Start pipeline", type="primary", use_container_width=True, disabled=not ready):
                result = api_post("/pipeline/start", start_payload(str(selected_source)))
                if result.get("error"): st.error(result["error"])
                else:
                    st.session_state.run_source = selected_source; st.success(result.get("message", "Pipeline started")); st.rerun()
        elif st.button("Stop pipeline", use_container_width=True):
            result = api_post("/pipeline/stop")
            if result.get("error"): st.error(result["error"])
            else: st.info(result.get("message", "Stop requested")); st.rerun()
        st.caption(f"API · {API_BASE}")
    with video:
        st.subheader("Annotated frame")
        st.markdown('<p class="note">Frame-level detector output used by the temporal filter.</p>', unsafe_allow_html=True)
        frame = live_frame_b64()
        if frame and status.get("source_state") != "idle":
            st.markdown(f'<div class="frame"><img src="data:image/jpeg;base64,{frame}" alt="Current annotated detector frame"></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="frame empty">No frame available. Select a source and start the pipeline.</div>', unsafe_allow_html=True)
        error = status.get("last_error")
        if error: st.error(f"{str(error.get('stage', 'pipeline')).replace('_', ' ').title()} failed · {error.get('message', 'Unknown runtime error')}")
        elif status.get("source_state") == "disconnected": st.warning("The source disconnected. This is a runtime state, not a nonviolent result.")
        strategy = status.get("temporal_strategy", setting("temporal_strategy", TEMPORAL_STRATEGY))
        required = int(status.get("frame_consistency", setting("frame_consistency", FRAME_CONSISTENCY)))
        release = int(status.get("negative_release_frames", setting("negative_release_frames", NEGATIVE_RELEASE_FRAMES)))
        with st.container(border=True):
            e1, e2, e3 = st.columns([1, 1.25, 1.25])
            e1.metric("Temporal state", temporal_phase(status))
            e2.metric("Window positives" if strategy == "rolling_window" else "Positive frames", f"{int(status.get('positive_run', 0) or 0)} / {required}")
            e3.metric("Negative release", f"{int(status.get('negative_run', 0) or 0)} / {release}")
    st.subheader("Current-run events")
    st.markdown('<p class="note">Only this run appears here. Persisted records remain in Event history.</p>', unsafe_allow_html=True)
    current = (api_get("/alerts/current", {"alerts": []}) or {"alerts": []}).get("alerts", [])
    if not current: st.info("No event has qualified in this run.")
    for event in current[:8]: render_event(event)
    if st.toggle("Auto-refresh every second", value=False): time.sleep(1); st.rerun()

elif page == "Event history":
    st.subheader("Persistent event history")
    st.markdown('<p class="note">Detected events and notification delivery are separate facts.</p>', unsafe_allow_html=True)
    events = load_event_history()
    if not events: st.info("No persisted event records are available.")
    else:
        known_dates = [value for event in events if (value := event_date(event))]
        first, last = min(known_dates, default=date.today()), max(known_dates, default=date.today())
        options = sorted({compact_source_label(e.get("source")) for e in events} | {str(e.get("location") or "Unknown") for e in events})
        f1, f2, f3 = st.columns([1.2, 1, 1])
        selected_dates = f1.date_input("Date range", value=(first, last), min_value=first, max_value=last)
        source_filter = f2.selectbox("Source or location", ["All"] + options)
        notification_filter = f3.selectbox("Notification outcome", ["All", "not_attempted", "queued", "accepted", "failed"], format_func=lambda value: value.replace("_", " ").title())
        date_range = tuple(selected_dates) if isinstance(selected_dates, (list, tuple)) and len(selected_dates) == 2 else None
        filtered = filter_history(events, dates=date_range, source_or_location=source_filter, notification=notification_filter)
        a1, a2 = st.columns([1, 4], vertical_alignment="center")
        a1.download_button("Export filtered CSV", events_as_csv(filtered), file_name="event_history.csv", mime="text/csv")
        a2.caption(f"{len(filtered)} of {len(events)} events")
        if not filtered: st.info("No events match these filters.")
        for event in filtered[:30]: render_event(event, divider=True)

else:
    st.subheader("Settings")
    st.markdown('<p class="note">Saved values apply to the next run. Supported changes also update an active run.</p>', unsafe_allow_html=True)
    with st.form("settings"):
        st.markdown("### Detection")
        confidence = st.slider("Confidence threshold (C)", .05, 1.0, float(setting("confidence", CONFIDENCE_THRESHOLD)), .05, help="Minimum detector confidence that counts as positive.")
        st.caption("C — minimum confidence required for a frame to count as positive.")
        st.markdown("### Temporal filtering")
        strategy = st.selectbox("Strategy", ["consecutive", "rolling_window"], index=0 if setting("temporal_strategy", TEMPORAL_STRATEGY) == "consecutive" else 1, format_func=lambda value: "Consecutive frames" if value == "consecutive" else "Rolling-window voting")
        positives = st.number_input("Positive observations required (N or M)", 1, 120, int(setting("frame_consistency", FRAME_CONSISTENCY)))
        st.caption("N — consecutive positives required. With rolling voting, M positives are required within W frames.")
        negatives = st.number_input("Negative release frames (K)", 1, 120, int(setting("negative_release_frames", NEGATIVE_RELEASE_FRAMES)))
        st.caption("K — consecutive negatives required to end the event and rearm.")
        with st.expander("Advanced temporal settings", expanded=False):
            window = st.number_input("Rolling window size (W)", int(positives), 120, max(int(positives), int(setting("rolling_window_size", ROLLING_WINDOW_SIZE))), disabled=strategy != "rolling_window")
        st.markdown("### Notifications")
        telegram = st.checkbox("Enable Telegram notifications", value=bool(setting("enable_telegram", ENABLE_TELEGRAM_ALERTS)))
        st.caption("Credentials remain in environment configuration and are never displayed here.")
        with st.expander("Advanced notification settings", expanded=False):
            cooldown = st.number_input("Accepted-notification cooldown (seconds)", 0, 86400, int(setting("cooldown", ALERT_COOLDOWN_SECONDS)))
        submitted = st.form_submit_button("Save settings", type="primary")
    if submitted:
        st.session_state.update(confidence=confidence, frame_consistency=int(positives), temporal_strategy=strategy, rolling_window_size=int(window), negative_release_frames=int(negatives), cooldown=int(cooldown), enable_telegram=telegram)
        if status.get("running"):
            result = api_post("/pipeline/config", {"confidence": confidence, "frame_consistency": int(positives), "temporal_strategy": strategy, "rolling_window_size": int(window), "negative_release_frames": int(negatives), "cooldown_seconds": int(cooldown), "enable_telegram": telegram})
            if result.get("error"): st.error(result["error"])
            else: st.success("Saved for the next run and applied to the active pipeline.")
        else: st.success("Saved. These values will apply when the next run starts.")
