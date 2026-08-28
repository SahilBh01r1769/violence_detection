"""Public Streamlit demo for the violence-detection project."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

# Streamlit Community Cloud executes this entrypoint with ``demo/`` as the
# script directory. Add the repository root explicitly so the demo can reuse
# the production-oriented root modules (config.py, core/, etc.) regardless of
# the process working directory.
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
    page_title="AI Violence Detection Demo",
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
AIRTLAB_SOURCE = (
    "https://github.com/airtlab/"
    "A-Dataset-for-Automatic-Violence-Detection-in-Videos"
)
PROJECT_SOURCE = (
    "https://github.com/SahilBh01r1769/violence_detection/"
    "tree/fix/pretrained-model-runtime"
)

MAX_UPLOAD_MB = 50


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
    st.subheader("Choose a video")
    mode = st.radio(
        "Demo source",
        ["Violent sample", "Non-violent sample", "Upload your video"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Upload your video":
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
            st.error(
                f"Video is {size_mb:.1f} MB. Please upload a file under {MAX_UPLOAD_MB} MB."
            )
            return None, None, None
        suffix = Path(uploaded.name).suffix.lower() or ".mp4"
        path = write_temp_video(data, suffix)
        st.caption(f"Selected: {uploaded.name} · {size_mb:.1f} MB")
        return path, uploaded.name, data

    sample_url = VIOLENT_SAMPLE_URL if mode == "Violent sample" else NORMAL_SAMPLE_URL
    sample_name = (
        "AIRTLab violent sample"
        if mode == "Violent sample"
        else "AIRTLab non-violent sample"
    )
    try:
        data = download_sample(sample_url)
    except Exception as exc:
        st.error(f"Could not download the public sample clip: {exc}")
        return None, None, None
    path = write_temp_video(data)
    st.caption(f"{sample_name} · research/educational sample")
    return path, sample_name, data


def render_results(summary) -> None:
    if summary.alerts_triggered > 0:
        verdict = "VIOLENCE DETECTED"
    elif summary.violent_samples > 0:
        verdict = "SUSPICIOUS ACTIVITY"
    else:
        verdict = "NO VIOLENCE DETECTED"

    st.divider()
    st.subheader("Analysis result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", verdict)
    c2.metric("Peak confidence", f"{summary.peak_confidence:.1%}")
    c3.metric("Analyzed frames", f"{summary.frames_analyzed:,}")
    c4.metric("Alerts", summary.alerts_triggered)

    st.caption(
        f"Video duration: {summary.duration_seconds:.1f}s · "
        f"Analyzed at ~{summary.analysis_fps:.1f} FPS · "
        f"Violent sampled frames: {summary.violent_samples}/{summary.frames_analyzed}"
    )

    if summary.output_video and summary.output_video.exists():
        st.subheader("Annotated output")
        st.video(str(summary.output_video))

    if summary.timeline:
        st.subheader("Detection timeline")
        timeline = pd.DataFrame(summary.timeline)
        chart = timeline[["time_s", "violent"]].set_index("time_s")
        st.area_chart(chart, height=180)
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
        st.subheader("Alert evidence")
        columns = st.columns(min(3, len(summary.alert_frames)))
        for idx, frame in enumerate(summary.alert_frames[:6]):
            with columns[idx % len(columns)]:
                st.image(
                    frame,
                    caption=f"Alert frame {idx + 1}",
                    use_container_width=True,
                    channels="BGR",
                )


st.title("AI Violence Detection")
st.markdown(
    "A hosted demonstration of a **YOLOv8-based video monitoring pipeline** with "
    "temporal consistency filtering and alert generation."
)

with st.sidebar:
    st.header("Demo settings")
    target_analysis_fps = st.slider(
        "Inference rate (frames/sec)",
        min_value=1,
        max_value=8,
        value=6,
        help="6 FPS is the tested demo default; lower values reduce CPU use but can change temporal alert behavior.",
    )
    max_duration = st.slider(
        "Maximum analyzed duration (seconds)",
        min_value=5,
        max_value=45,
        value=30,
        step=5,
    )
    st.info(
        "Email and WhatsApp delivery are intentionally disabled in the public demo. "
        "The full project retains those integrations."
    )
    st.markdown(f"[View production-oriented branch]({PROJECT_SOURCE})")


tab_demo, tab_architecture = st.tabs(["Demo", "Architecture & attribution"])

with tab_demo:
    video_path, source_name, source_bytes = source_picker()

    if source_bytes:
        st.subheader("Input preview")
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
            progress.progress(
                fraction,
                text=f"Analyzing video... {done}/{total} sampled frames",
            )
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
        except Exception as exc:
            progress.empty()
            status.empty()
            st.exception(exc)

    if "last_summary" in st.session_state:
        if st.session_state.get("last_source"):
            st.caption(f"Results for: {st.session_state['last_source']}")
        render_results(st.session_state["last_summary"])

with tab_architecture:
    st.subheader("Production project architecture")
    st.code(
        """CCTV / RTSP / local video
        ↓
FastAPI backend
        ↓
YOLOv8 violence detector
        ↓
N-frame temporal consistency filter
        ↓
Alert manager
        ├── screenshot/history
        ├── email
        └── WhatsApp
        ↓
Streamlit monitoring dashboard""",
        language="text",
    )

    st.subheader("Hosted demo architecture")
    st.code(
        """Sample video / uploaded clip
        ↓
Streamlit public demo
        ↓
Same core ViolenceDetector
        ↓
YOLOv8 inference
        ↓
Same temporal consistency logic
        ↓
Annotated result + timeline + demo alerts""",
        language="text",
    )

    st.markdown(
        """
**What is original project work**

- Real-time frame processing and model integration
- Temporal consistency filtering across consecutive frames
- FastAPI service layer for the production-oriented runtime
- Streamlit monitoring, alert history and analytics
- Screenshot persistence and notification integrations
- Docker/configuration/deployment integration

**Model attribution**

The hosted demo uses the same third-party pretrained YOLO violence-detection weights
documented in the project. The model weights are not presented as original training work.

**Sample attribution**

The two built-in demo clips come from the AIRTLab *Dataset for Automatic Violence
Detection in Videos*, released for research and educational use.
"""
    )
    st.markdown(f"[AIRTLab dataset source]({AIRTLAB_SOURCE})")
