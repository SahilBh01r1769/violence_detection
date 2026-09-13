"""Standalone replay UI used by the disposable demonstration branch."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
CLIPS = {
    "Violent clip · boxing": {
        "video": str(ASSETS / "violent_annotated.mp4"),
        "frame": str(ASSETS / "violent_event.jpg"),
        "source": "violence_indoor_pexels_6296646.mp4",
        "location": "Camera-01",
        "frames": 312,
        "elapsed": "00:12",
        "events": 1,
        "phase": "Active",
        "positive": "5 / 5",
        "negative": "0 / 3",
        "confidence": 0.84,
        "notification": "Accepted",
        "time": "2026-09-11 21:38:14",
    },
    "Nonviolent clip · street scene": {
        "video": str(ASSETS / "nonviolent_annotated.mp4"),
        "frame": None,
        "source": "non_violence_daylight_pexels_33567764.mp4",
        "location": "Walkway",
        "frames": 395,
        "elapsed": "00:13",
        "events": 0,
        "phase": "Inactive",
        "positive": "0 / 5",
        "negative": "0 / 3",
        "confidence": 0.0,
        "notification": "Not attempted",
        "time": "—",
    },
}

st.set_page_config(page_title="Temporal Violence Events", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
<style>
:root{--page:#151A1D;--panel:#1B2125;--raised:#20272B;--border:#3B474D;--text:#E7ECEF;--muted:#A6B0B5;--amber:#D79A4A;--red:#C95B5B;--green:#6F9B88;--blue:#7897AE}
html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:var(--page);color:var(--text);font-family:Inter,Arial,sans-serif}
[data-testid="stAppViewContainer"] .block-container{max-width:1320px;padding:1.2rem 2rem 3rem}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none}h1,h2,h3,label,p,span,div{color:var(--text)}
h1{font-size:1.55rem!important;margin:0!important;letter-spacing:-.02em}h2{font-size:1.2rem!important;margin-top:1.5rem!important}h3{font-size:1rem!important}
.kicker{color:var(--amber);font-size:.72rem;letter-spacing:.11em;text-transform:uppercase}.note{color:var(--muted);font-size:.82rem;margin:-.55rem 0 .75rem}
hr{border-color:var(--border)!important}[data-testid="stHorizontalBlock"]{align-items:stretch}
[data-testid="stMetric"]{background:transparent;border:0;padding:.15rem .7rem}[data-testid="stMetricLabel"] p{color:var(--muted)!important;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em}[data-testid="stMetricValue"]{font-size:1.05rem}
[data-testid="stVerticalBlockBorderWrapper"],[data-testid="stForm"],[data-testid="stExpander"]{background:var(--panel);border-color:var(--border)!important;border-radius:6px;box-shadow:none}
.frame{background:#090C0E;border:1px solid var(--border);border-radius:5px;padding:8px;overflow:hidden}.meta{color:var(--muted);font-size:.82rem;line-height:1.55}
.chip{display:inline-block;border:1px solid var(--green);color:var(--green);border-radius:3px;padding:.17rem .45rem;font-size:.72rem;font-weight:650}
[data-testid="stAlert"]{border-radius:4px;border:1px solid var(--border);background:var(--raised)}
.stButton>button,.stDownloadButton>button,[data-baseweb="select"]>div,[data-testid="stTextInputRootElement"],[data-testid="stNumberInputContainer"]{background:var(--raised)!important;color:var(--text)!important;border-radius:4px!important;box-shadow:none!important;border-color:var(--border)!important}
[data-baseweb="popover"],[data-baseweb="popover"]>div,[role="listbox"],[role="option"],[data-baseweb="menu"]{background:#252D31!important;color:var(--text)!important}
[role="option"]:hover,[aria-selected="true"]{background:#344047!important}[role="option"] span,[role="option"] div{color:var(--text)!important}
input,textarea,[data-baseweb="select"] span,[data-baseweb="select"] svg{color:var(--text)!important;fill:var(--text)!important}
.stButton>button[kind="primary"]{background:var(--amber)!important;color:#101417!important;border-color:var(--amber)!important;font-weight:700}
[data-testid="stVideo"] video{width:100%;max-height:610px;background:#090C0E}
@media(max-width:760px){[data-testid="stAppViewContainer"] .block-container{padding:1rem}}
</style>
""", unsafe_allow_html=True,
)


def header() -> str:
    title, nav = st.columns([1.05, 1.35], vertical_alignment="center")
    with title:
        st.markdown('<div class="kicker">Applied ML systems experiment</div>', unsafe_allow_html=True)
        st.title("Temporal Violence Events")
    with nav:
        return st.radio("View", ["Monitor", "Event history", "Settings"], horizontal=True, label_visibility="collapsed")


def event_details(selected: dict) -> None:
    image, details = st.columns([1.65, 1])
    with image:
        st.image(selected["frame"], use_container_width=True)
    with details:
        with st.container(border=True):
            st.markdown("**Event #1 · Violence**")
            st.markdown(
                '<div class="meta">'
                f"Confidence&nbsp;&nbsp; {selected['confidence']:.0%}<br>"
                f"Time&nbsp;&nbsp; {selected['time']}<br>"
                f"Location&nbsp;&nbsp; {selected['location']}<br>"
                f"Source&nbsp;&nbsp; {selected['source']}</div>",
                unsafe_allow_html=True,
            )
            st.caption("Notification outcome")
            st.write(selected["notification"])


page = header()
choice = st.session_state.get("demo_choice", list(CLIPS)[0])
selected = CLIPS[choice]

if page == "Monitor":
    with st.container(border=True):
        s1, s2, s3, s4, s5 = st.columns([1.05, 1.55, 1, 1, 1])
        s1.markdown('<span class="chip">Connected</span>', unsafe_allow_html=True)
        s2.metric("Source", selected["source"])
        s3.metric("Frames", f"{selected['frames']:,}")
        s4.metric("Events", selected["events"])
        s5.metric("Elapsed", selected["elapsed"])

    controls, video = st.columns([.30, .70], gap="large")
    with controls:
        st.subheader("Source")
        choice = st.selectbox("Source type", list(CLIPS), index=list(CLIPS).index(choice))
        st.session_state.demo_choice = choice
        selected = CLIPS[choice]
        st.text_input("Location label", value=selected["location"], disabled=True)
        c1, c2 = st.columns(2)
        c1.button("Start pipeline", type="primary", use_container_width=True)
        c2.button("Stop", use_container_width=True)
    with video:
        st.subheader("Annotated frame")
        st.markdown('<p class="note">Frame-level detector output used by the temporal filter.</p>', unsafe_allow_html=True)
        st.video(selected["video"], autoplay=True, muted=True, loop=True)
        with st.container(border=True):
            e1, e2, e3 = st.columns([1, 1.25, 1.25])
            e1.metric("Temporal state", selected["phase"])
            e2.metric("Positive frames", selected["positive"])
            e3.metric("Negative release", selected["negative"])

    st.subheader("Current-run events")
    st.markdown('<p class="note">Only this run appears here. Persisted records remain in Event history.</p>', unsafe_allow_html=True)
    if selected["frame"]:
        event_details(selected)
    else:
        st.info("No event has qualified in this run.")

elif page == "Event history":
    st.subheader("Persistent event history")
    st.markdown('<p class="note">Detected events and notification delivery are separate facts.</p>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1.2, 1, 1])
    f1.date_input("Date range", value=())
    f2.selectbox("Source or location", ["All", "Camera-01", "violence_indoor_pexels_6296646.mp4"])
    f3.selectbox("Notification outcome", ["All", "Accepted", "Not attempted", "Failed"])
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "timestamp", "class", "confidence", "location", "source", "notification_status"])
    writer.writerow([1, CLIPS[list(CLIPS)[0]]["time"], "violence", 0.84, "Camera-01", CLIPS[list(CLIPS)[0]]["source"], "accepted"])
    st.download_button("Export filtered CSV", output.getvalue(), "event_history.csv", "text/csv")
    event_details(CLIPS[list(CLIPS)[0]])

else:
    st.subheader("Settings")
    st.markdown('<p class="note">Saved values apply when the next run starts.</p>', unsafe_allow_html=True)
    with st.form("settings"):
        st.markdown("### Detection")
        st.slider("Confidence threshold (C)", .05, 1.0, .70, .05)
        st.caption("C — minimum confidence required for a frame to count as positive.")
        st.markdown("### Temporal filtering")
        strategy = st.selectbox("Strategy", ["Consecutive frames", "Rolling-window voting"])
        st.number_input("Positive observations required (N or M)", 1, 120, 5)
        st.caption("N — consecutive positives required. With rolling voting, M positives are required within W frames.")
        st.number_input("Negative release frames (K)", 1, 120, 3)
        st.caption("K — consecutive negatives required to end the event and rearm.")
        with st.expander("Advanced temporal settings", expanded=False):
            st.number_input("Rolling window size (W)", 5, 120, 7, disabled=strategy == "Consecutive frames")
        st.markdown("### Notifications")
        st.checkbox("Enable Telegram notifications", value=True)
        st.caption("Credentials remain in environment configuration and are never displayed here.")
        with st.expander("Advanced notification settings", expanded=False):
            st.number_input("Accepted-notification cooldown (seconds)", 0, 86400, 30)
        st.form_submit_button("Save settings", type="primary")
