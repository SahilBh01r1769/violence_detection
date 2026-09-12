"""Disposable hosted showcase for the Violence Detection dashboard."""

from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
CLIPS = {
    "Violent clip · boxing": {
        "video": "https://videos.pexels.com/video-files/6296646/6296646-sd_960_406_25fps.mp4",
        "frame": ASSETS / "violent_event.jpg",
        "frames": 312,
        "fps": 25.0,
        "events": 1,
        "event": "Active",
        "positive": "5 / 5",
        "negative": "0 / 3",
        "caption": "Qualified event after five consecutive positive observations.",
        "source": "Pexels · men playing boxing · 6296646",
    },
    "Nonviolent clip · street scene": {
        "video": "https://videos.pexels.com/video-files/33567764/14271282_960_540_60fps.mp4",
        "frame": None,
        "frames": 395,
        "fps": 30.0,
        "events": 0,
        "event": "Idle",
        "positive": "0 / 5",
        "negative": "0 / 3",
        "caption": "No event qualified in this run.",
        "source": "Pexels · people walking under trees · 33567764",
    },
}

st.set_page_config(page_title="Temporal Violence Events", layout="wide")
st.markdown(
    """
<style>
:root { --page:#f1eee8; --panel:#e7e1d7; --line:#bdb4a8; --ink:#29251f; --muted:#6b645b; --accent:#82503a; }
html, body, [data-testid="stAppViewContainer"] { background:var(--page); color:var(--ink); font-family:Arial,Helvetica,sans-serif; }
[data-testid="stSidebar"] { background:#e8e2d8; border-right:1px solid var(--line); }
[data-testid="stAppViewContainer"] .block-container { max-width:1180px; padding-top:2rem; }
h1,h2,h3 { color:var(--ink)!important; letter-spacing:-.02em; }
.hero { background:linear-gradient(135deg,#2f3e46 0%,#50616d 62%,#82503a 100%); color:#fff; padding:22px 26px; border-radius:10px; margin-bottom:20px; }
.hero h1 { color:#fff!important; margin:0 0 6px; font-size:2rem; }
.hero p { margin:0; max-width:780px; color:#eee9e2; }
.section-note { color:var(--muted); margin-top:-8px; margin-bottom:14px; }
.event-card { background:var(--panel); border:1px solid var(--line); border-left:5px solid var(--accent); border-radius:7px; padding:14px 16px; margin-bottom:10px; }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:3px; padding:12px; box-shadow:none; }
[data-baseweb="select"] > div, .stButton > button { border-radius:3px!important; box-shadow:none!important; }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Temporal Violence Events")
    st.radio("Navigation", ["Pipeline", "Event History", "Settings"], index=0)
    st.subheader("Input")
    choice = st.selectbox("Demo clip", list(CLIPS))
    selected = CLIPS[choice]
    st.caption(f"Source: {selected['source']}")
    st.divider()
    st.button("Start", type="primary", use_container_width=True)
    st.button("Stop", use_container_width=True)
    st.caption("ACTIVE")

st.markdown(
    """
    <div class="hero">
      <h1>Temporal Violence Events</h1>
      <p>Frame-level model predictions are qualified into persistent events for review and alerting.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Source", "Connected")
m2.metric("Frames", f"{selected['frames']:,}")
m3.metric("Current events", selected["events"])
m4.metric("FPS", f"{selected['fps']:.1f}")

st.subheader("Current frame")
st.markdown(
    '<p class="section-note">Bounding boxes show the checkpoint class and confidence. This model knows only violence and non-violence classes.</p>',
    unsafe_allow_html=True,
)
st.video(selected["video"])

st.subheader("Temporal state")
t1, t2, t3, t4 = st.columns(4)
t1.metric("Event", selected["event"])
t2.metric("Positive run", selected["positive"])
t3.metric("Negative run", selected["negative"])
t4.metric("Cooldown", "0s")
st.caption("Decision confidence ≥ 0.70. Five consecutive positive frames start an event; three negative frames end it.")

st.subheader("Events from this run")
if selected["frame"]:
    st.markdown(
        '<div class="event-card"><strong>Event #1 · violence</strong><br>84% confidence</div>',
        unsafe_allow_html=True,
    )
    st.image(str(selected["frame"]), width=640, caption=selected["caption"])
else:
    st.info(selected["caption"])

st.caption("Dashboard showcase running the recorded detection output at normal video speed.")
