# Real-Time Violence Detection & Alert System

A video-monitoring application that combines a **pretrained YOLOv8 fight/violence detector** with OpenCV, temporal filtering, FastAPI, Streamlit, Email alerts, and Twilio WhatsApp alerts.

The project does **not require model training** to run. On first use it can download a public pretrained violence/fight checkpoint, while `utils/train.py` remains available only for users who want to train their own compatible model.

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
- Four-page Streamlit dashboard: Live Monitor, Alert History, Analytics, Settings
- CLI/headless operation
- Docker Compose support
- Optional training helper

## Default pretrained model

The default checkpoint is the YOLOv8-nano model published by **Musawer1214/Fight-Violence-detection-yolov8**:

https://github.com/Musawer1214/Fight-Violence-detection-yolov8

The upstream project documents two classes, **Violence/Fight** and **NoViolence/NoFight**, and documents class ID `1` as Violence/Fight. The checkpoint is not committed to this repository; it is downloaded directly from the upstream project on first use.

See [`THIRD_PARTY_MODELS.md`](THIRD_PARTY_MODELS.md) for attribution and limitations.

> The pretrained model was not trained by this repository's author. This repository contributes the real-time video pipeline, temporal filtering, API, dashboard, alert orchestration, persistence, configuration, and deployment integration around the model.

## Project structure

```text
violence_detection/
├── alerts/
│   ├── alert_manager.py
│   ├── email_alert.py
│   └── whatsapp_alert.py
├── api/
│   └── server.py
├── core/
│   ├── detector.py
│   ├── pipeline.py
│   └── stream.py
├── dashboard/
│   └── app.py
├── tests/
├── utils/
│   ├── download_model.py
│   ├── logger.py
│   └── train.py
├── config.py
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

Runtime directories such as `models/`, `screenshots/`, and `logs/` are created as needed and excluded from Git.

## Quick start

### 1. Install

```bash
git clone https://github.com/SahilBh01r1769/violence_detection.git
cd violence_detection
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
# Linux/macOS
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Email and Twilio credentials are optional. A missing notification configuration does not stop the detection pipeline.

### 3. Download the model

The detector downloads the default model automatically when it is missing. You can also download it explicitly:

```bash
python -m utils.download_model
```

Default path:

```text
models/violence_yolov8n.pt
```

To use another compatible YOLO checkpoint, configure `MODEL_PATH`, `VIOLENCE_CLASS_IDS`, and/or `VIOLENCE_CLASSES` in `.env`.

## Run the application

Start the API:

```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Start the dashboard in another terminal:

```bash
streamlit run dashboard/app.py --server.port 8501
```

Open:

- Dashboard: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`

The dashboard Start button launches the detection pipeline using the source and settings configured on the Settings page.

### Headless CLI

Webcam:

```bash
python -m core.pipeline --source 0 --location "Camera-01" --display
```

Video file:

```bash
python -m core.pipeline --source path/to/video.mp4 --location "Test-Video" --display
```

RTSP camera:

```bash
python -m core.pipeline --source "rtsp://user:password@camera/stream" --location "Entrance"
```

## Detection logic

For every frame:

1. YOLO produces bounding boxes, class IDs, labels, and confidence scores.
2. A detection is considered violent when its class ID or normalized class name matches the configured violence classes.
3. A frame is considered violent when at least one violent detection is present.
4. The result is added to a sliding temporal window.
5. An alert event is accepted only when all `FRAME_CONSISTENCY` frames in that window are violent.
6. The cooldown is checked before an alert screenshot or notification is generated.

Alert metadata is selected from violent detections only, so a higher-confidence non-violent box cannot become the alert label.

## Dashboard

### Live Monitor
Displays the latest annotated frame, pipeline state, frame count, alert count, FPS, confidence, and cooldown.

### Alert History
Displays persisted alert records with class/confidence filtering, screenshot viewing, and CSV export.

### Analytics
Displays alert counts, class distribution, confidence distribution, and alerts over time.

### Settings
Controls:

- video source
- camera/location label
- confidence threshold
- consecutive-frame requirement
- alert cooldown
- Email notification enable/disable
- WhatsApp notification enable/disable

Detection settings and notification toggles can be changed while the pipeline is running. Source/location changes take effect on the next start.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | API health check |
| `GET` | `/status` | Pipeline status and runtime settings |
| `GET` | `/alerts` | Paginated persisted alert history |
| `GET` | `/alerts/{id}/screenshot` | Retrieve an alert screenshot |
| `POST` | `/pipeline/start` | Start video processing |
| `POST` | `/pipeline/stop` | Stop video processing |
| `POST` | `/pipeline/config` | Update runtime detection/notification settings |
| `GET` | `/stream/frame` | Latest annotated JPEG frame |

## Notifications

### Email

Set in `.env`:

```text
SMTP_USER=
SMTP_PASSWORD=
ALERT_RECIPIENTS=
```

For Gmail, use an App Password rather than your normal account password.

### WhatsApp

Set:

```text
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_FROM=
TWILIO_WHATSAPP_TO=
```

The current WhatsApp integration sends text details. Twilio cannot attach a local screenshot path; a public media URL would be required for image media.

## Docker

Create `.env` first, then run:

```bash
docker compose up --build
```

The container exposes ports `8000` and `8501`, persists `models/`, `screenshots/`, and `logs/` through bind mounts, and uses `/health` for its container healthcheck.

Docker webcam passthrough is platform-specific. On Linux, uncomment the `/dev/video0` `devices` block in `docker-compose.yml`. RTSP and video-file sources do not require that mapping.

## Tests

Install development requirements and run:

```bash
pip install -r requirements-dev.txt
pytest -q
```

The focused tests cover violence-class selection, alert metadata, temporal-window reconfiguration, webcam source parsing, API validation, and runtime settings updates. They do not claim to measure the pretrained model's accuracy.

## Optional training

**Training is not required for the default setup.**

`utils/train.py` remains available if you later obtain a YOLO-format labelled dataset and GPU access:

```bash
python -m utils.train --data dataset.yaml --model yolov8m.pt --device 0 --validate
```

If a custom model uses a different class scheme, update `VIOLENCE_CLASS_IDS` / `VIOLENCE_CLASSES` accordingly.

## AWS usage

AWS is optional. An EC2 GPU instance can be useful for temporary benchmarking, model validation, or a hosted demo, but the repository does not require AWS or retraining before use.

## Limitations

- The default checkpoint is a third-party violence/fight detector; this repository does not claim ownership of its training or accuracy.
- Violence recognition is per-frame YOLO inference plus an N-consecutive-frame heuristic, not a learned video-temporal model.
- False positives and false negatives are possible and depend on the upstream model, scene, camera angle, lighting, and threshold.
- The dashboard/API do not include authentication and should not be exposed directly to the public internet without an authentication/reverse-proxy layer.
- Surveillance deployments must follow applicable privacy, consent, and data-retention requirements.

## Privacy and intended use

This project is intended for learning, prototyping, and authorized monitoring environments. Alert screenshots and history are stored locally. Configured Email/Twilio channels send alert information to those external services when enabled.
