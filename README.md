# Real-Time Violence Detection & Alert System

A video-monitoring application that combines a **pretrained YOLOv8 fight/violence detector** with OpenCV, temporal filtering, FastAPI, Streamlit, SMTP Email alerts, and Twilio WhatsApp alerts.

**Training is not required.** The repository uses a public pretrained violence/fight checkpoint and keeps custom training as an optional utility only.

## Features

- Webcam, RTSP/IP-camera, and video-file input
- YOLOv8 violence/fight inference
- Configurable confidence threshold
- N-consecutive-frame temporal consistency filter
- Alert cooldown and annotated screenshots
- SMTP Email alerts
- Twilio WhatsApp alerts
- Persistent local alert history
- FastAPI control/status/history endpoints
- Streamlit dashboard: Live Monitor, Alert History, Analytics, Settings
- CLI/headless operation
- Docker Compose support
- Automated tests on Python 3.11
- Optional custom-model training helper

## Requirements

- **Python 3.11** recommended
- Internet access for the first model download
- A webcam, RTSP stream, or local video file for inference
- Email/Twilio credentials only if those notification channels are enabled

A GPU is optional. Ultralytics/PyTorch will use available hardware automatically; CPU execution is also supported.

## Default pretrained model

The default checkpoint is the YOLOv8-nano fight/violence model published by:

**Musawer1214/Fight-Violence-detection-yolov8**

https://github.com/Musawer1214/Fight-Violence-detection-yolov8

The upstream project documents two classes, `non_violence` and `violence`, with **class ID 1 representing violence**. This repository pins the download to upstream commit:

```text
20f0d05054cff7da2dc78dee3c2de1bd54106a13
```

The model is downloaded to `models/violence_yolov8n.pt` and is intentionally not committed to this repository.

See [`THIRD_PARTY_MODELS.md`](THIRD_PARTY_MODELS.md) for attribution and limitations.

> The pretrained checkpoint was not trained by this repository's author. This project contributes the real-time video pipeline, temporal filtering, API, dashboard, alerts, persistence, runtime configuration, and deployment integration around the model.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/SahilBh01r1769/violence_detection.git
cd violence_detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows PowerShell
venv\Scripts\Activate.ps1

# Windows Command Prompt
venv\Scripts\activate.bat

# Linux/macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create configuration

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux/macOS
cp .env.example .env
```

The default `.env.example` is sufficient for local detection. Email and WhatsApp are optional; if you do not intend to configure them, set:

```text
ENABLE_EMAIL_ALERTS=false
ENABLE_WHATSAPP_ALERTS=false
```

### 5. Download the pretrained model

Download it explicitly before starting the application:

```bash
python -m utils.download_model
```

Expected file:

```text
models/violence_yolov8n.pt
```

The application can also download it automatically if it is missing.

### 6. Start the API

From the repository root:

```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Check:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

### 7. Start the dashboard

Open a second terminal in the same repository and activate the same virtual environment:

```bash
streamlit run dashboard/app.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

Go to **Settings**, choose the video source, then press **Start** in the sidebar.

## Video sources

### Webcam

Use:

```text
0
```

or run directly:

```bash
python -m core.pipeline --source 0 --location "Camera-01" --display
```

### Video file

Use a path such as:

```text
C:\videos\sample.mp4
```

or:

```bash
python -m core.pipeline --source "path/to/video.mp4" --location "Test-Video" --display
```

Local video files stop cleanly at end-of-file; they are not looped automatically.

### RTSP/IP camera

Use the full RTSP URL:

```bash
python -m core.pipeline --source "rtsp://user:password@camera/stream" --location "Entrance"
```

Credentials in RTSP URLs are redacted from application log messages.

## Detection logic

For each frame:

1. YOLO returns bounding boxes, class IDs, labels, and confidence scores.
2. A detection is violent when its class ID or normalized class name matches the configured violence classes.
3. A frame is violent when at least one violent detection exists.
4. The result enters a temporal window.
5. An alert becomes eligible only when all `FRAME_CONSISTENCY` frames are violent.
6. The alert cooldown is checked before a screenshot or notification is generated.
7. Alert metadata is taken from the highest-confidence **violent** detection.

If model inference itself fails, the pipeline stops and logs the error instead of incorrectly treating the frame as safe.

## Dashboard

### Live Monitor
Shows the latest annotated frame, runtime state, frame count, alerts, FPS, confidence, consistency threshold, and cooldown.

### Alert History
Loads the complete persisted alert history through API pagination, supports class/confidence filtering, screenshot viewing, and CSV export.

### Analytics
Shows total alerts, average confidence, class counts, confidence distribution, and alerts over time.

### Settings
Controls:

- video source
- camera/location label
- confidence threshold
- consecutive-frame requirement
- alert cooldown
- Email alerts on/off
- WhatsApp alerts on/off

The dashboard initializes from `.env`. Confidence, consistency, cooldown, and notification toggles can be updated while the pipeline is running. Video source and location apply on the next start.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | API health check |
| `GET` | `/status` | Pipeline status and current runtime settings |
| `GET` | `/alerts` | Paginated persisted alert history |
| `GET` | `/alerts/{id}/screenshot` | Retrieve an alert screenshot |
| `POST` | `/pipeline/start` | Start video processing |
| `POST` | `/pipeline/stop` | Stop video processing |
| `POST` | `/pipeline/config` | Update runtime settings |
| `GET` | `/stream/frame` | Latest annotated JPEG frame |

The API waits for the previous processing thread to shut down before allowing a new pipeline to take control of the video source.

## Email alerts

Set these values in `.env`:

```text
ENABLE_EMAIL_ALERTS=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_RECIPIENTS=recipient@example.com
```

For Gmail, use a Google App Password rather than your normal account password.

## WhatsApp alerts

Set:

```text
ENABLE_WHATSAPP_ALERTS=true
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_FROM=whatsapp:+14155238886
TWILIO_WHATSAPP_TO=
```

The current integration sends text alert details. A local screenshot path cannot be attached directly through Twilio; image media requires a public HTTPS URL.

## Docker

Create `.env` first, then run:

```bash
docker compose up --build
```

Services:

```text
FastAPI:   http://localhost:8000
Streamlit: http://localhost:8501
```

The Compose setup persists:

- `models/`
- `screenshots/`
- `logs/`

and checks container health through `/health` using Python's standard library.

For a Linux host webcam, uncomment the `/dev/video0` device mapping in `docker-compose.yml`. RTSP inputs do not require webcam passthrough.

## Tests

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run:

```bash
pytest -q
```

Tests cover:

- violent-class selection
- alert metadata selection
- temporal-window reconfiguration
- video-source normalization
- API input validation
- runtime settings updates
- inference failure handling
- RTSP credential redaction
- stopped-runtime statistics

GitHub Actions runs the test suite on Python 3.11 for pushes to `main` and pull requests.

These tests validate application behavior; they do **not** claim or estimate the third-party model's accuracy.

## Optional custom training

Training is **not required** for the default project.

If you later have a YOLO-format labelled dataset and want a custom model:

```bash
python -m utils.train --data dataset.yaml --model yolov8m.pt --device 0 --validate
```

For another pretrained/custom checkpoint, update `MODEL_PATH`, `VIOLENCE_CLASS_IDS`, and/or `VIOLENCE_CLASSES` in `.env` to match that model's classes.

## Project structure

```text
violence_detection/
├── alerts/
├── api/
├── core/
├── dashboard/
├── tests/
├── utils/
├── .github/workflows/tests.yml
├── .dockerignore
├── .env.example
├── config.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Runtime model weights, screenshots, logs, local environments, and `.env` secrets are excluded from Git and Docker build context.

## Limitations

- The default checkpoint is a third-party violence/fight detector; this repository does not claim ownership of its training or accuracy.
- Detection is per-frame YOLO inference combined with an N-frame consistency heuristic, not a learned temporal video model.
- False positives and false negatives remain possible.
- The API/dashboard do not include authentication and should not be exposed directly to an untrusted public network.
- Surveillance usage must follow applicable privacy, consent, and retention requirements.

## Intended use

This project is intended for learning, prototyping, demonstrations, and authorized monitoring environments. Alert screenshots and history are stored locally. Configured Email/Twilio channels send alert information to those external services only when enabled.
