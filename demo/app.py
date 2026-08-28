"""Public Streamlit demo for the violence-detection project."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import requests
import streamlit as st

from config import (
    CONFIDENCE_THRESHOLD,
    FRAME_CONSISTENCY,
    MODEL_PATH,
    VIOLENCE_CLASSES,
    VIOLENCE_CLASS_IDS,
)
from core.detector import ViolenceDetector
from demo.processor import analyse_video

st.set_page_config(
    page_title="VisionGuard | Violence Detection Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

VIOLENT_SAMPLE_URL = (
    "https://raw.githubusercontent.com/airtlab/"
    "A-Dataset-for-Automatic-Violence-Detection-in-Videos/master/"
    "violence-detection-dataset/violent/cam1/8.mp4"
)
NORMAL_SAMPLE_URL = (
    "https://raw.githubusercontent.com/airtlab/"
    "A-Dataset-for-Automatic-Violence-Detection-in-Videos/master/"
    "violence-detection-dataset/non-violent/cam1/1.mp4"
)
AIRTLAB_SOURCE = "https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos"
PROJECT_SOURCE = "https://github.com/SahilBh01r1769/violence_detection"
MAX_UPLOAD_MB = 50


st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(148,163,184,.14);}
.vg-hero {
    padding: 1.55rem 1.65rem;
    border: 1px solid rgba(148,163,184,.16);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(20,27,45,.96), rgba(11,16,32,.96));
    margin-bottom: 1rem;
}
.vg-eyebrow {font-size:.78rem; letter-spacing:.18em; font-weight:700; color:#f87171;}
.vg-title {font-size:2.25rem; line-height:1.05; font-weight:800; margin:.35rem 0 .45rem 0;}
.vg-subtitle {color:#aeb8cc; font-size:1.03rem; max-width:760px;}
.vg-pills {display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1rem;}
.vg-pill {padding:.32rem .62rem; border-radius:999px; border:1px solid rgba(148,163,184,.18); background:rgba(148,163,184,.07); font-size:.75rem; color:#dbe3f0;}
.vg-section-title {font-size:1.15rem; font-weight:700; margin: .2rem 0 .7rem 0;}
.vg-card {padding:1rem 1.05rem; border:1px solid rgba(148,163,184,.15); border-radius:14px; background:rgba(20,27,45,.76); min-height:100%;}
.vg-card h4 {margin:0 0 .25rem 0; font-size:1rem;}
.vg-muted {color:#9da8bb; font-size:.88rem;}
.vg-verdict {padding:1.25rem; border-radius:16px; border:1px solid rgba(148,163,184,.15); text-align:center; margin:.65rem 0 1rem 0;}
.vg-danger {background:rgba(127,29,29,.30); border-color:rgba(248,113,113,.38);}
.vg-warn {background:rgba(120,53,15,.30); border-color:rgba(251,146,60,.35);}
.vg-safe {background:rgba(20,83,45,.30); border-color:rgba(74,222,128,.32);}
.vg-verdict-label {font-size:1.55rem; font-weight:800; letter-spacing:.02em;}
.vg-verdict-sub {color:#bac4d5; margin-top:.25rem;}
div[data-testid="stMetric"] {background:rgba(20,27,45,.72); border:1px solid rgba(148,163,184,.14); padding:.85rem; border-radius:14px;}
.stButton > button {border-radius:10px; font-weight:700; min-height:2.9rem;}
hr {border-color:rgba(148,163,184,.14) !important;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading YOLO violence-detection model...")
def get_detector() -> ViolenceDetector:
    return ViolenceDetector(
        model_path=MODEL_PATH,
        confidence=CONFIDENCE_THRESHOLD,
        frame_consistency=FRAME_CONSISTENCY,
        violence_classes=VIOLENCE_CLASSES,
        violence_class_ids=VIOLENCE_CLASS_IDS,
    )


@st.cache_data(show_spinner=False)
def download_sample(url: str) -> bytes:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def write_temp_video(data: bytes, suffix: str = ".mp4") -> Path:
    digest = hashlib.sha256(data).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"violence_demo_{digest}{suffix}"
    if not path.exists() or path.stat().st_size != len(data):
        path.write_bytes(data)
    return path


def source_picker() -> tuple[Path | None, str | None, bytes | None]:
    st.markdown('<div class="vg-section-title">Choose a video source</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Demo source",
        ["Violence sample", "Normal sample", "Upload video"],
        horizontal=True,
        label_visibility="collapsed",
    )

    helper = {
        "Violence sample": "Run the alert path using a public research sample.",
        "Normal sample": "Run a negative control and confirm no alert is triggered.",
        "Upload video": "Analyze your own MP4, MOV or AVI clip (up to 50 MB).",
    }
    st.caption(helper[mode])

    if mode == "Upload video":
        uploaded = st.file_uploader(
            "Upload MP4, MOV or AVI",
            type=["mp4", "mov", "avi"],
            help=f"Maximum recommended size: {MAX_UPLOAD_MB} MB.",
        )
        if uploaded is None:
            return None, None, None
        data = uploaded.getvalue()
        size_mb = len(data) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            st.error(f"Video is {size_mb:.1f} MB. Please upload a file under {MAX_UPLOAD_MB} MB.")
            return None, None, None
        suffix = Path(uploaded.name).suffix.lower() or ".mp4"
        return write_temp_video(data, suffix), uploaded.name, data

    sample_url = VIOLENT_SAMPLE_URL if mode == "Violence sample" else NORMAL_SAMPLE_URL
    sample_name = "AIRTLab violence sample" if mode == "Violence sample" else "AIRTLab normal sample"
    try:
        data = download_sample(sample_url)
    except Exception as exc:
        st.error(f"Could not download the public sample clip: {exc}")
        return None, None, None
    return write_temp_video(data), sample_name, data


def verdict_html(summary) -> str:
    if summary.alerts_triggered > 0:
        css, verdict, detail = "vg-danger", "VIOLENCE DETECTED", "Temporal alert condition confirmed"
    elif summary.violent_samples > 0:
        css, verdict, detail = "vg-warn", "SUSPICIOUS ACTIVITY", "Violent frames detected, but alert threshold was not confirmed"
    else:
        css, verdict, detail = "vg-safe", "NO VIOLENCE DETECTED", "No alert threshold reached in the analyzed frames"
    return f"""
<div class="vg-verdict {css}">
  <div class="vg-verdict-label">{verdict}</div>
  <div class="vg-verdict-sub">{detail}</div>
</div>
"""


def render_results(summary, source_bytes: bytes | None) -> None:
    st.markdown(verdict_html(summary), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak confidence", f"{summary.peak_confidence:.1%}")
    c2.metric("Analyzed frames", f"{summary.frames_analyzed:,}")
    c3.metric("Violent frames", f"{summary.violent_samples:,}")
    c4.metric("Alerts", summary.alerts_triggered)

    st.caption(
        f"Analyzed {summary.duration_seconds:.1f}s at ~{summary.analysis_fps:.1f} sampled FPS · "
        f"Configured threshold {CONFIDENCE_THRESHOLD:.0%} · Temporal window {FRAME_CONSISTENCY} frames"
    )

    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="vg-section-title">Input video</div>', unsafe_allow_html=True)
        if source_bytes:
            st.video(source_bytes)
        else:
            st.info("Input preview is unavailable after this rerun.")
    with right:
        st.markdown('<div class="vg-section-title">Detection output</div>', unsafe_allow_html=True)
        if summary.output_video and summary.output_video.exists():
            st.video(str(summary.output_video))
        else:
            st.info("Annotated video output is unavailable for this clip.")

    if summary.timeline:
        st.divider()
        st.markdown('<div class="vg-section-title">Confidence timeline</div>', unsafe_allow_html=True)
        timeline = pd.DataFrame(summary.timeline)
        chart = timeline[["time_s", "confidence", "violent"]].copy().set_index("time_s")
        chart["violent"] = chart["violent"].astype(float)
        st.line_chart(chart, height=230)
        st.caption("Confidence is model confidence; the violence flag is 1 when a sampled frame is classified as violent.")

        with st.expander("Frame-level analysis data"):
            st.dataframe(
                timeline.rename(
                    columns={
                        "time_s": "Time (s)",
                        "violent": "Violence flag",
                        "confidence": "Confidence",
                        "class_name": "Predicted class",
                        "alert": "Alert triggered",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    if summary.alert_frames:
        st.divider()
        st.markdown('<div class="vg-section-title">Alert evidence</div>', unsafe_allow_html=True)
        st.caption("Annotated frames captured when the temporal alert condition was met.")
        columns = st.columns(min(3, len(summary.alert_frames)))
        for idx, frame in enumerate(summary.alert_frames[:6]):
            with columns[idx % len(columns)]:
                st.image(frame, caption=f"Alert frame {idx + 1}", use_container_width=True, channels="BGR")


st.markdown(
    """
<div class="vg-hero">
  <div class="vg-eyebrow">VISIONGUARD · HOSTED DEMO</div>
  <div class="vg-title">AI Violence Detection</div>
  <div class="vg-subtitle">Analyze a sample or uploaded clip with the same YOLOv8 detector and temporal consistency logic used by the full project.</div>
  <div class="vg-pills">
    <span class="vg-pill">PRETRAINED YOLOV8</span>
    <span class="vg-pill">TEMPORAL FILTERING</span>
    <span class="vg-pill">REAL INFERENCE</span>
    <span class="vg-pill">CPU-FRIENDLY DEMO</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Analysis controls")
    target_analysis_fps = st.slider(
        "Inference rate (frames/sec)",
        min_value=1,
        max_value=8,
        value=6,
        help="Lower values reduce CPU usage; higher values sample the source more frequently.",
    )
    max_duration = st.slider(
        "Maximum analyzed duration (seconds)",
        min_value=5,
        max_value=45,
        value=30,
        step=5,
    )
    st.divider()
    st.markdown("### Model")
    st.caption("YOLOv8 Nano · pretrained violence/fight detector")
    st.caption(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    st.caption(f"Temporal window: {FRAME_CONSISTENCY} frames")
    st.divider()
    st.info("Email, WhatsApp, webcam and RTSP controls are intentionally omitted from the public hosted demo.")
    st.markdown(f"[View full project on GitHub]({PROJECT_SOURCE})")


tab_demo, tab_about = st.tabs(["Interactive demo", "Architecture & attribution"])

with tab_demo:
    video_path, source_name, source_bytes = source_picker()

    if source_bytes and "last_summary" not in st.session_state:
        st.markdown('<div class="vg-section-title">Preview</div>', unsafe_allow_html=True)
        st.video(source_bytes)

    analyze = st.button(
        "Analyze video",
        type="primary",
        disabled=video_path is None,
        use_container_width=True,
    )

    if analyze and video_path is not None:
        progress = st.progress(0, text="Preparing analysis...")
        status = st.empty()

        def on_progress(done: int, total: int) -> None:
            fraction = 0.0 if total <= 0 else min(1.0, done / total)
            progress.progress(fraction, text=f"Analyzing video... {done}/{total} sampled frames")
            status.caption("Running real YOLO inference on sampled video frames.")

        try:
            detector = get_detector()
            with st.spinner("Running violence detection..."):
                summary = analyse_video(
                    video_path,
                    detector,
                    target_analysis_fps=float(target_analysis_fps),
                    max_duration_seconds=float(max_duration),
                    progress_callback=on_progress,
                )
            progress.progress(1.0, text="Analysis complete")
            status.empty()
            st.session_state["last_summary"] = summary
            st.session_state["last_source"] = source_name
            st.session_state["last_source_bytes"] = source_bytes
        except Exception as exc:
            progress.empty()
            status.empty()
            st.error("Analysis failed. The demo stops on inference errors rather than treating failed frames as safe.")
            st.exception(exc)

    if "last_summary" in st.session_state:
        st.divider()
        if st.session_state.get("last_source"):
            st.caption(f"Results for: {st.session_state['last_source']}")
        render_results(st.session_state["last_summary"], st.session_state.get("last_source_bytes"))

with tab_about:
    st.markdown("### How the hosted demo fits the project")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### Full project")
        st.code(
            """Webcam / RTSP / video file
        ↓
OpenCV video pipeline
        ↓
YOLOv8 violence detector
        ↓
N-frame temporal consistency
        ↓
Alert manager + persistence
        ↓
FastAPI + Streamlit dashboard
        ↓
Email / WhatsApp alerts""",
            language="text",
        )
    with col2:
        st.markdown("#### Hosted demo")
        st.code(
            """Sample / uploaded video
        ↓
Streamlit hosted app
        ↓
Same ViolenceDetector
        ↓
YOLOv8 inference
        ↓
Same temporal consistency
        ↓
Annotated video + timeline""",
            language="text",
        )

    with st.expander("What is original project work", expanded=True):
        st.markdown(
            """
- Video ingestion and frame-processing pipeline
- Integration of the pretrained detector into application runtime
- Consecutive-frame temporal filtering and alert behavior
- FastAPI service layer and Streamlit monitoring UI in the full project
- Screenshot/history persistence and Email/WhatsApp integrations
- Docker, configuration, testing and deployment integration
"""
        )

    with st.expander("Model attribution"):
        st.markdown(
            "The demo uses the same third-party pretrained YOLO violence/fight weights documented in the main project. "
            "The checkpoint is not presented as original model-training work."
        )
        st.markdown(f"[View the full repository]({PROJECT_SOURCE})")

    with st.expander("Sample attribution"):
        st.markdown(
            "The two built-in clips come from the AIRTLab *Dataset for Automatic Violence Detection in Videos* "
            "and are used here as research/educational demo samples."
        )
        st.markdown(f"[AIRTLab dataset source]({AIRTLAB_SOURCE})")
