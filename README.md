# 🛡️ Real-Time Violence Detection & Alert System

**Author:** Sahil Bhoir | **Stack:** YOLOv8 · OpenCV · FastAPI · Streamlit · Twilio · SMTP

An end-to-end pipeline that watches a webcam/RTSP/video-file stream, runs YOLOv8 inference
on each frame, requires **N consecutive violent frames** before firing an alert (to suppress
false positives), and dispatches Email + WhatsApp notifications with a cooldown so you don't
get flooded. Includes a FastAPI backend and a 4-page Streamlit dashboard.

> **Verification note:** This README was checked against the actual source in this repo —
> every module was read, imported, and exercised (detector inference, temporal-consistency
> window, alert cooldown + dispatch + history persistence, and all FastAPI endpoints) before
> writing these instructions. Two real bugs were found and are documented below with fixes.

---

## 📁 Project Structure

```
violence-detection/
│
├── config.py                   # Central config — reads from .env
├── requirements.txt
├── .env.example                 # Copy to .env and fill credentials
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh
│
├── core/
│   ├── detector.py              # YOLOv8 inference + temporal consistency window
│   ├── stream.py                 # VideoCapture wrapper + screenshot saving
│   └── pipeline.py               # Main loop — stream → detect → alert
│
├── alerts/
│   ├── email_alert.py            # HTML email via SMTP
│   ├── whatsapp_alert.py         # WhatsApp via Twilio
│   └── alert_manager.py          # Cooldown + dispatch orchestration
│
├── api/
│   └── server.py                  # FastAPI REST backend
│
├── dashboard/
│   └── app.py                     # Streamlit 4-page dashboard
│
├── utils/
│   ├── logger.py                  # Rotating file + console logging
│   └── train.py                   # YOLOv8 fine-tuning helper
│
├── models/                       # Place your .pt weights here (not tracked in git)
├── screenshots/                  # Auto-saved alert screenshots (auto-created)
└── logs/                         # Application + alert history logs (auto-created)
```

`models/`, `screenshots/`, and `logs/` don't need to be created manually — `config.py`
creates `screenshots/` and `logs/` on import, and Docker creates all three at build time.

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/SahilBh01r1769/violence_detection.git
cd violence_detection

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

Key variables to set:

| Variable                                   | Description                                   |
| ------------------------------------------- | ---------------------------------------------- |
| `MODEL_PATH`                                | Path to your fine-tuned `.pt` weights          |
| `VIDEO_SOURCE`                              | `0` for webcam, or `rtsp://...` for IP camera  |
| `SMTP_USER` / `SMTP_PASSWORD`               | Gmail address + App Password                   |
| `ALERT_RECIPIENTS`                          | Comma-separated alert email addresses          |
| `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN`  | From Twilio console                            |
| `TWILIO_WHATSAPP_TO`                        | Recipient WhatsApp: `whatsapp:+91XXXXXXXXXX`   |

Every channel degrades gracefully: if SMTP or Twilio credentials are missing, that
channel is skipped and logged as a warning instead of crashing the pipeline (verified).

### 3. Add Your Model

Place your fine-tuned YOLOv8 weights at the path set in `MODEL_PATH`
(default: `models/violence_yolov8.pt`).

> If no model file is found at that path, the system **automatically falls back to
> pretrained `yolov8n.pt`** (downloaded on first run). This is a stock COCO model — it
> detects `person`, `car`, etc., **not violence** — so `is_violent`/alerting will never
> meaningfully trigger until you train and place your own weights (see below).

---

## 🚀 Running the System

### Option A — Run components separately (development)

```bash
# Terminal 1: Start the FastAPI backend
python -m uvicorn api.server:app --reload --port 8000

# Terminal 2: Start the Streamlit dashboard
streamlit run dashboard/app.py --server.port 8501

# Terminal 3: (Optional) Run the pipeline headlessly from the CLI
python -m core.pipeline --source 0 --location "Camera-01" --display
```

> ⚠️ **Run `core/pipeline.py` and `utils/train.py` as modules (`python -m core.pipeline`,
> `python -m utils.train`), not as plain scripts.** Both files do `from config import ...`
> / `from utils.logger import ...`, which relies on the repo root being on `sys.path`.
> Python only adds the *script's own folder* (`core/` or `utils/`) to `sys.path` when you
> run `python core/pipeline.py` directly — so that form fails immediately with
> `ModuleNotFoundError: No module named 'config'`. This was confirmed by running both
> forms against the actual code. Always invoke from the **repo root** using `-m`.

Both `api/server.py` (via `uvicorn` import-string) and `dashboard/app.py` (via
`streamlit run`) are unaffected by this — they were tested and start correctly as
documented.

### Option B — Docker (production)

```bash
docker-compose up --build
```

- Dashboard: <http://localhost:8501>
- API docs: <http://localhost:8000/docs>

> **Known issue:** the `docker-compose.yml` healthcheck runs `curl -f http://localhost:8000/health`,
> but the Dockerfile's runtime stage never installs `curl`. The container will run fine,
> but `docker ps` will show the healthcheck itself failing/erroring rather than reporting
> real health. Fix by adding `curl` to the runtime stage's `apt-get install` list, or by
> switching the healthcheck to Python: `python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"`.

The compose file mounts `/dev/video0` for a local webcam — remove that `devices:` block if
you're using an RTSP/IP camera instead, or it will fail to start on machines without a
`/dev/video0` device.

---

## 🎓 Training Your Own Model

### 1. Prepare Dataset

Export from Roboflow in **YOLOv8 format** or structure your own:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Create `dataset.yaml`:

```yaml
path: /path/to/dataset
train: images/train
val:   images/val
test:  images/test

nc: 4
names: ['Normal', 'Fighting', 'Weapon', 'Aggression']
```

Class names **must match** `VIOLENCE_CLASSES` in `config.py` (`Fighting`, `Weapon`,
`Aggression` by default) — the detector only treats a box as violent if `class_name`
is in that set, so a mismatched label set will silently never alert.

### 2. Fine-tune

```bash
python -m utils.train \
  --data    dataset/dataset.yaml \
  --model   yolov8m.pt \
  --epochs  50 \
  --imgsz   640 \
  --batch   16 \
  --device  0 \
  --output  models/violence_yolov8.pt \
  --validate
```

Use `--device cpu` if you don't have a CUDA GPU.

### 3. Recommended Datasets

- [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) — 2,000 surveillance clips
- [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) — real CCTV footage
- [Roboflow Universe](https://universe.roboflow.com) — search "violence", "fighting", "weapon"

---

## 🔔 Notification Setup

### Email (Gmail)

1. Enable 2FA on your Google account
2. Generate an **App Password**: Google Account → Security → App Passwords
3. Set `SMTP_USER` and `SMTP_PASSWORD` in `.env`

Sends a styled HTML email with an inline screenshot attachment (`email_alert.py`) —
confirmed to build and send correctly given valid credentials, and to fail gracefully
(logged, returns `False`, doesn't crash the pipeline) without them.

### WhatsApp (Twilio Sandbox)

1. Sign up at [twilio.com](https://www.twilio.com)
2. Go to **Messaging → Try it out → Send a WhatsApp message**
3. Follow sandbox join instructions (send "join &lt;code&gt;" to the sandbox number)
4. Set credentials in `.env`

For production WhatsApp, request a Twilio WhatsApp Business number. Note: Twilio can't
attach your local screenshot file directly — `whatsapp_alert.py` only attaches media if
you separately host the screenshot and pass a public `media_url`; otherwise it sends the
text alert with the local filename mentioned for reference.

### Alert Cooldown & Dedup

`AlertManager` enforces a single global cooldown (`ALERT_COOLDOWN_SECONDS`) across both
channels — confirmed: a second `trigger()` call inside the cooldown window is suppressed
and returns `False`, while dispatch itself happens in a background thread so the video
loop is never blocked waiting on SMTP/Twilio.

---

## 🌐 API Reference

All endpoints below were smoke-tested against a running instance of `api/server.py`.

| Method | Endpoint                  | Description                          |
| ------ | -------------------------- | -------------------------------------- |
| GET    | `/health`                  | Health check                          |
| GET    | `/status`                  | Pipeline stats (FPS, alerts, uptime)  |
| GET    | `/alerts`                  | Alert history (paginated)             |
| GET    | `/alerts/{id}/screenshot`  | Fetch screenshot image                |
| POST   | `/pipeline/start`          | Start detection                       |
| POST   | `/pipeline/stop`           | Stop detection                        |
| POST   | `/pipeline/config`         | Update confidence / cooldown          |
| GET    | `/stream/frame`            | Latest annotated JPEG frame           |

Full interactive docs: <http://localhost:8000/docs>

`/stream/frame` returns a black "No stream available" placeholder JPEG when no pipeline
is running yet, rather than erroring — confirmed via test client.

---

## ⚙️ Configuration Reference

| Parameter                | Default | Description                             |
| -------------------------- | --------- | ------------------------------------------ |
| `CONFIDENCE_THRESHOLD`     | `0.55`  | Minimum detection confidence            |
| `FRAME_CONSISTENCY`        | `5`     | Consecutive violent frames before alert |
| `ALERT_COOLDOWN_SECONDS`   | `30`    | Min time between alerts                 |
| `FPS_TARGET`                | `20`    | Target processing frame rate            |
| `MAX_SCREENSHOTS`          | `500`   | Auto-purge threshold                    |

**Detection mechanism, in detail:** each frame runs through YOLO inference; any box whose
class name is in `VIOLENCE_CLASSES` marks that frame as violent. A `deque(maxlen=FRAME_CONSISTENCY)`
sliding window records violent/not-violent per frame. An alert only fires once the window
is **completely full of violent frames** (`len(window) == FRAME_CONSISTENCY and all(window)`),
at which point the window is cleared so the same sustained event doesn't re-fire every
subsequent frame. This was traced through the code and confirmed with synthetic frames.

---

## 🔒 Privacy & Ethics

- This system is intended for **authorized surveillance only**
- Obtain proper consent before monitoring any space
- Implement role-based access control for the dashboard in production
- Consider adding face blurring (planned future feature) for GDPR compliance
- All alert data is stored locally — no third-party data sharing beyond configured channels

---

## 🗺️ Future Enhancements

- [ ] Face blurring / anonymisation
- [ ] Edge deployment (Raspberry Pi, NVIDIA Jetson)
- [ ] Mobile push notifications
- [ ] Audio analysis (screaming, gunshot detection)
- [ ] Multi-camera grid view
- [ ] ONVIF / RTSP CCTV integration
- [ ] Fix the known issues listed above
