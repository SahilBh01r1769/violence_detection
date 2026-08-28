"""Streamlit dashboard for live monitoring, alert history, analytics and settings."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
st.set_page_config(page_title="Violence Detection System", page_icon="🛡️", layout="wide")


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
        if response.ok:
            return data
        return {"error": data.get("detail") or data.get("message") or response.text}
    except Exception as exc:
        return {"error": str(exc)}


def live_frame_b64() -> str | None:
    try:
        response = requests.get(f"{API_BASE}/stream/frame", timeout=3)
        response.raise_for_status()
        return base64.b64encode(response.content).decode()
    except Exception:
        return None


def load_alert_history() -> pd.DataFrame:
    data = api_get("/alerts?per_page=100", {"alerts": []}) or {"alerts": []}
    alerts = data.get("alerts", [])
    if not alerts:
        return pd.DataFrame()
    frame = pd.DataFrame(alerts)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame


def setting(name: str, default):
    return st.session_state.get(name, default)


def start_payload() -> dict:
    return {
        "source": str(setting("video_source", "0")),
        "location": setting("location", "Camera-01"),
        "confidence": float(setting("confidence", 0.55)),
        "frame_consistency": int(setting("frame_consistency", 5)),
        "cooldown_seconds": int(setting("cooldown", 30)),
        "enable_email": bool(setting("enable_email", True)),
        "enable_whatsapp": bool(setting("enable_whatsapp", True)),
    }


with st.sidebar:
    st.title("🛡️ Violence Detection")
    page = st.radio("Navigation", ["Live Monitor", "Alert History", "Analytics", "Settings"])
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start", type="primary", use_container_width=True):
            result = api_post("/pipeline/start", start_payload())
            st.error(result["error"]) if result.get("error") else st.success(result.get("message", "Pipeline started"))
    with c2:
        if st.button("Stop", use_container_width=True):
            result = api_post("/pipeline/stop")
            st.error(result["error"]) if result.get("error") else st.info(result.get("message", "Stop signal sent"))
    status = api_get("/status", {}) or {}
    st.caption("ACTIVE" if status.get("running") else "OFFLINE")
    st.caption(f"API: {API_BASE}")


if page == "Live Monitor":
    st.header("Live Monitor")
    status = api_get("/status", {}) or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", "Active" if status.get("running") else "Offline")
    m2.metric("Frames", f"{status.get('frames_processed', 0):,}")
    m3.metric("Alerts", status.get("alerts_fired", 0))
    m4.metric("FPS", f"{status.get('fps', 0):.1f}")
    frame_b64 = live_frame_b64()
    if frame_b64:
        st.markdown(f'<img src="data:image/jpeg;base64,{frame_b64}" style="width:100%;max-width:960px">', unsafe_allow_html=True)
    else:
        st.info("No frame available. Start the API and pipeline first.")
    st.write(
        f"Cooldown remaining: **{status.get('cooldown_remaining', 0):.0f}s** · "
        f"Confidence: **{status.get('confidence', setting('confidence', 0.55)):.2f}** · "
        f"Frame consistency: **{status.get('frame_consistency', setting('frame_consistency', 5))}**"
    )
    if st.checkbox("Auto-refresh every second", value=True):
        time.sleep(1)
        st.rerun()

elif page == "Alert History":
    st.header("Alert History")
    df = load_alert_history()
    if df.empty:
        st.info("No alert records available yet.")
    else:
        classes = ["All"] + sorted(df["detected_class"].dropna().unique().tolist())
        c1, c2 = st.columns(2)
        class_filter = c1.selectbox("Class", classes)
        min_confidence = c2.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
        filtered = df[df["confidence"] >= min_confidence].copy()
        if class_filter != "All":
            filtered = filtered[filtered["detected_class"] == class_filter]
        st.dataframe(filtered.drop(columns=["screenshot_path"], errors="ignore"), use_container_width=True, hide_index=True)
        st.download_button("Export CSV", filtered.drop(columns=["screenshot_path"], errors="ignore").to_csv(index=False), file_name="alert_history.csv", mime="text/csv")
        st.subheader("Screenshots")
        for _, row in filtered.head(20).iterrows():
            path_value = row.get("screenshot_path")
            if path_value and Path(str(path_value)).exists():
                st.image(str(path_value), caption=f"Alert #{int(row['id'])} — {row['detected_class']}")

elif page == "Analytics":
    st.header("Analytics")
    df = load_alert_history()
    if df.empty:
        st.info("Analytics will appear after alerts are recorded.")
    else:
        a, b, c = st.columns(3)
        a.metric("Total alerts", len(df))
        b.metric("Average confidence", f"{df['confidence'].mean():.1%}")
        c.metric("Most common class", df["detected_class"].mode().iloc[0])
        class_counts = df["detected_class"].value_counts().rename_axis("class").reset_index(name="count")
        st.plotly_chart(px.bar(class_counts, x="class", y="count", title="Alerts by class"), use_container_width=True)
        st.plotly_chart(px.histogram(df, x="confidence", nbins=20, title="Confidence distribution"), use_container_width=True)
        timeline = df.dropna(subset=["timestamp"]).copy()
        if not timeline.empty:
            timeline["hour"] = timeline["timestamp"].dt.floor("h")
            counts = timeline.groupby("hour").size().reset_index(name="count")
            st.plotly_chart(px.line(counts, x="hour", y="count", markers=True, title="Alerts over time"), use_container_width=True)

elif page == "Settings":
    st.header("Settings")
    status = api_get("/status", {}) or {}
    with st.form("settings"):
        confidence = st.slider("Confidence threshold", 0.05, 1.0, float(setting("confidence", 0.55)), 0.05)
        frame_consistency = st.number_input("Consecutive violent frames required", 1, 120, int(setting("frame_consistency", 5)))
        cooldown = st.number_input("Alert cooldown (seconds)", 0, 86400, int(setting("cooldown", 30)))
        location = st.text_input("Camera/location label", setting("location", "Camera-01"))
        video_source = st.text_input("Video source", str(setting("video_source", "0")))
        enable_email = st.checkbox("Enable Email alerts", value=bool(setting("enable_email", True)))
        enable_whatsapp = st.checkbox("Enable WhatsApp alerts", value=bool(setting("enable_whatsapp", True)))
        submitted = st.form_submit_button("Save & Apply")
    if submitted:
        st.session_state.update(confidence=confidence, frame_consistency=int(frame_consistency), cooldown=int(cooldown), location=location, video_source=video_source, enable_email=enable_email, enable_whatsapp=enable_whatsapp)
        if status.get("running"):
            result = api_post("/pipeline/config", {"confidence": confidence, "frame_consistency": int(frame_consistency), "cooldown_seconds": int(cooldown), "enable_email": enable_email, "enable_whatsapp": enable_whatsapp})
            st.error(result["error"]) if result.get("error") else st.success("Settings applied to the running pipeline.")
        else:
            st.success("Settings saved. They will be used the next time the pipeline starts.")
    st.caption("Changing the video source or location takes effect on the next pipeline start.")
