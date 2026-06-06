"""
dashboard/app.py — Streamlit Monitoring Dashboard

Pages:
  1. Live Monitor  — real-time feed + detection status
  2. Alert History — searchable table with screenshots
  3. Analytics     — charts: alerts over time, class breakdown
  4. Settings      — configure confidence, cooldown, notifications
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Violence Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Dark theme overrides */
  [data-testid="stAppViewContainer"] { background: #0d0d0d; }
  [data-testid="stSidebar"]          { background: #141414; border-right: 1px solid #2a2a2a; }
  .block-container                   { padding-top: 1.5rem; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 14px 18px;
  }

  /* Alert badge */
  .alert-badge {
    display: inline-block;
    background: #c0392b;
    color: white;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .5px;
  }
  .safe-badge {
    display: inline-block;
    background: #1e8449;
    color: white;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 700;
  }

  /* Section headers */
  .section-header {
    font-size: 18px;
    font-weight: 700;
    color: #e0e0e0;
    border-left: 4px solid #e74c3c;
    padding-left: 10px;
    margin-bottom: 16px;
  }

  /* Status dot */
  .dot-green { color: #2ecc71; font-size: 18px; }
  .dot-red   { color: #e74c3c; font-size: 18px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def api_get(path: str, default=None):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default


def api_post(path: str, payload: dict = None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload or {}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_live_frame_b64() -> str | None:
    try:
        r = requests.get(f"{API_BASE}/stream/frame", timeout=3)
        if r.status_code == 200:
            return base64.b64encode(r.content).decode()
    except Exception:
        pass
    return None


def load_alert_history() -> pd.DataFrame:
    data = api_get("/alerts?per_page=200", default={"alerts": []})
    alerts = data.get("alerts", [])
    if not alerts:
        return pd.DataFrame()
    df = pd.DataFrame(alerts)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=60)
    st.markdown("### 🛡️ Violence Detection")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🎥 Live Monitor", "🚨 Alert History", "📊 Analytics", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick pipeline controls
    st.markdown("**Pipeline Control**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            result = api_post("/pipeline/start", {
                "source":   st.session_state.get("video_source", "0"),
                "location": st.session_state.get("location", "Camera-01"),
            })
            st.toast(result.get("message", "Done"))
    with col2:
        if st.button("⏹ Stop", use_container_width=True):
            result = api_post("/pipeline/stop")
            st.toast(result.get("message", "Done"))

    st.markdown("---")
    status = api_get("/status", default={})
    running = status.get("running", False)
    if running:
        st.markdown('<span class="dot-green">●</span> **System ACTIVE**', unsafe_allow_html=True)
    else:
        st.markdown('<span class="dot-red">●</span> **System OFFLINE**', unsafe_allow_html=True)

    st.caption(f"API: {API_BASE}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

if page == "🎥 Live Monitor":
    st.markdown('<div class="section-header">Live Monitor</div>', unsafe_allow_html=True)

    status = api_get("/status", default={})

    # ── KPI Row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status",           "🟢 Active" if status.get("running") else "🔴 Offline")
    c2.metric("Frames Processed", f'{status.get("frames_processed", 0):,}')
    c3.metric("Alerts Fired",     status.get("alerts_fired", 0))
    c4.metric("Live FPS",         f'{status.get("fps", 0):.1f}')

    st.markdown("---")

    # ── Live Feed ─────────────────────────────────────────────────────────────
    feed_col, info_col = st.columns([3, 1])

    with feed_col:
        st.markdown("**Live Feed**")
        feed_placeholder = st.empty()

    with info_col:
        st.markdown("**Detection Info**")
        info_placeholder = st.empty()
        st.markdown("**Cooldown**")
        cooldown_bar = st.empty()
        st.markdown("**Recent Alerts**")
        recent_placeholder = st.empty()

    # ── Auto-refresh loop ────────────────────────────────────────────────────
    auto_refresh = st.checkbox("Auto-refresh (1s)", value=True)

    if auto_refresh:
        # Refresh every second
        frame_b64 = get_live_frame_b64()
        if frame_b64:
            feed_placeholder.markdown(
                f'<img src="data:image/jpeg;base64,{frame_b64}" style="width:100%;border-radius:8px;border:1px solid #333;">',
                unsafe_allow_html=True,
            )
        else:
            feed_placeholder.info("No live feed. Start the pipeline and ensure your camera is connected.")

        # Status info
        cooldown = status.get("cooldown_remaining", 0)
        cooldown_bar.progress(
            min(1.0, cooldown / max(1, status.get("cooldown_seconds", 30))),
            text=f"Cooldown: {cooldown:.0f}s",
        )

        # Recent alerts
        df = load_alert_history()
        if not df.empty:
            recent = df.head(5)[["timestamp", "detected_class", "confidence"]]
            recent["confidence"] = recent["confidence"].map(lambda x: f"{x:.0%}")
            recent_placeholder.dataframe(recent, hide_index=True, use_container_width=True)
        else:
            recent_placeholder.caption("No alerts yet.")

        info_placeholder.markdown(f"""
        | Field | Value |
        |---|---|
        | Uptime | {status.get('uptime_seconds', 0):.0f}s |
        | Frames | {status.get('frames_processed', 0):,} |
        | Alerts | {status.get('alerts_fired', 0)} |
        """)

        time.sleep(1)
        st.rerun()
    else:
        feed_placeholder.info("Enable auto-refresh to see the live feed.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ALERT HISTORY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🚨 Alert History":
    st.markdown('<div class="section-header">Alert History</div>', unsafe_allow_html=True)

    df = load_alert_history()

    if df.empty:
        st.info("No alerts recorded yet. The system will log every detection event here.")
    else:
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            classes = ["All"] + sorted(df["detected_class"].unique().tolist())
            cls_filter = st.selectbox("Filter by Class", classes)
        with fc2:
            min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
        with fc3:
            show_screenshots = st.checkbox("Show Screenshots", value=True)

        # Apply filters
        filtered = df.copy()
        if cls_filter != "All":
            filtered = filtered[filtered["detected_class"] == cls_filter]
        filtered = filtered[filtered["confidence"] >= min_conf]

        st.caption(f"Showing {len(filtered)} of {len(df)} alerts")

        for _, row in filtered.iterrows():
            with st.expander(
                f"🚨  Alert #{int(row['id'])}  |  {row['detected_class']}  "
                f"|  {row['confidence']:.0%}  |  {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}",
                expanded=False,
            ):
                ec1, ec2 = st.columns([1, 2])
                with ec1:
                    st.markdown(f"**Class:** `{row['detected_class']}`")
                    st.markdown(f"**Confidence:** `{row['confidence']:.1%}`")
                    st.markdown(f"**Location:** `{row.get('location', 'N/A')}`")
                    st.markdown(f"**Timestamp:** `{row['timestamp']}`")
                    st.markdown(
                        f"**Email:** {'✅' if row.get('email_sent') else '❌'}  "
                        f"**WhatsApp:** {'✅' if row.get('whatsapp_sent') else '❌'}"
                    )
                with ec2:
                    if show_screenshots and row.get("screenshot_path"):
                        p = Path(row["screenshot_path"])
                        if p.exists():
                            st.image(str(p), caption="Screenshot at detection", use_column_width=True)
                        else:
                            st.caption("Screenshot file not found.")

        # Export
        st.markdown("---")
        csv = filtered.drop(columns=["screenshot_path"], errors="ignore").to_csv(index=False)
        st.download_button(
            "⬇️ Export as CSV",
            data=csv,
            file_name=f"alert_history_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Analytics":
    st.markdown('<div class="section-header">Analytics</div>', unsafe_allow_html=True)

    df = load_alert_history()

    if df.empty:
        st.info("No alert data yet. Analytics will appear once detections are recorded.")
    else:
        # ── Summary KPIs ──────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Alerts",       len(df))
        k2.metric("Avg Confidence",      f'{df["confidence"].mean():.1%}')
        k3.metric("Most Common Class",   df["detected_class"].mode()[0])
        k4.metric("Email Success Rate",  f'{df["email_sent"].mean():.0%}')

        st.markdown("---")

        # ── Alerts Over Time ──────────────────────────────────────────────────
        st.markdown("**Alerts Over Time**")
        df["hour"] = df["timestamp"].dt.floor("H")
        time_data  = df.groupby("hour").size().reset_index(name="count")
        fig_time = px.bar(
            time_data, x="hour", y="count",
            color_discrete_sequence=["#e74c3c"],
            template="plotly_dark",
        )
        fig_time.update_layout(
            paper_bgcolor="#1a1a1a", plot_bgcolor="#1a1a1a",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="", yaxis_title="Alerts",
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # ── Class Distribution + Confidence Distribution ──────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Class Distribution**")
            class_data = df["detected_class"].value_counts().reset_index()
            class_data.columns = ["Class", "Count"]
            fig_pie = px.pie(
                class_data, names="Class", values="Count",
                color_discrete_sequence=px.colors.sequential.Reds_r,
                template="plotly_dark",
            )
            fig_pie.update_layout(
                paper_bgcolor="#1a1a1a",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            st.markdown("**Confidence Distribution**")
            fig_hist = px.histogram(
                df, x="confidence", nbins=20,
                color_discrete_sequence=["#e67e22"],
                template="plotly_dark",
            )
            fig_hist.update_layout(
                paper_bgcolor="#1a1a1a", plot_bgcolor="#1a1a1a",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Confidence", yaxis_title="Count",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Notification Success ──────────────────────────────────────────────
        st.markdown("**Notification Success Rate**")
        notif_data = pd.DataFrame({
            "Channel": ["Email", "WhatsApp"],
            "Success": [df["email_sent"].sum(), df["whatsapp_sent"].sum()],
            "Failed":  [(~df["email_sent"]).sum(), (~df["whatsapp_sent"]).sum()],
        })
        fig_notif = px.bar(
            notif_data, x="Channel", y=["Success", "Failed"],
            barmode="group",
            color_discrete_map={"Success": "#2ecc71", "Failed": "#e74c3c"},
            template="plotly_dark",
        )
        fig_notif.update_layout(
            paper_bgcolor="#1a1a1a", plot_bgcolor="#1a1a1a",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_notif, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Settings":
    st.markdown('<div class="section-header">Settings</div>', unsafe_allow_html=True)

    with st.form("pipeline_settings"):
        st.markdown("**Detection Parameters**")
        sc1, sc2 = st.columns(2)
        with sc1:
            confidence = st.slider(
                "Confidence Threshold",
                0.1, 1.0,
                float(st.session_state.get("confidence", 0.55)),
                0.05,
                help="Minimum confidence score to register a detection",
            )
            frame_consistency = st.number_input(
                "Frame Consistency (frames)",
                1, 30,
                int(st.session_state.get("frame_consistency", 5)),
                help="Number of consecutive violent frames before alert fires",
            )
        with sc2:
            cooldown = st.number_input(
                "Alert Cooldown (seconds)",
                5, 300,
                int(st.session_state.get("cooldown", 30)),
                help="Minimum seconds between successive alerts",
            )
            location = st.text_input(
                "Camera / Location Label",
                st.session_state.get("location", "Camera-01"),
            )

        st.markdown("---")
        st.markdown("**Video Source**")
        video_source = st.text_input(
            "Source (0 = webcam, or RTSP URL)",
            st.session_state.get("video_source", "0"),
        )

        st.markdown("---")
        st.markdown("**Notifications**")
        nc1, nc2 = st.columns(2)
        with nc1:
            enable_email = st.checkbox("Enable Email Alerts", value=True)
        with nc2:
            enable_whatsapp = st.checkbox("Enable WhatsApp Alerts", value=True)

        submitted = st.form_submit_button("💾 Save & Apply", type="primary")

    if submitted:
        st.session_state.update({
            "confidence":        confidence,
            "frame_consistency": frame_consistency,
            "cooldown":          cooldown,
            "location":          location,
            "video_source":      video_source,
        })

        # Push runtime config to API
        api_post("/pipeline/config", {
            "confidence":       confidence,
            "cooldown_seconds": cooldown,
        })

        st.success("Settings saved! Restart the pipeline for source/location changes to take effect.")

    # ── .env Preview ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Environment Variables (.env)**")
    st.caption("Copy this into your .env file with the real credentials filled in.")
    env_path = Path(".env.example")
    if env_path.exists():
        st.code(env_path.read_text(), language="bash")
    else:
        st.info(".env.example not found in working directory.")
