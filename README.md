# 🛡️ Real-Time Violence Detection & Alert System

**Author:** Sahil Bhoir | **Date:** June 2026  
**Stack:** YOLOv8 · OpenCV · FastAPI · Streamlit · Twilio · SMTP

---

## 📁 Project Structure

```
violence-detection/
│
├── config.py                   # Central config — reads from .env
├── requirements.txt
├── .env.example                # Copy to .env and fill credentials
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh
│
├── core/
│   ├── detector.py             # YOLOv8 inference + temporal consistency
│   ├── stream.py               # VideoCapture wrapper + screenshot saving
│   └── pipeline.py             # Main loop — stream → detect → alert
│
├── alerts/
│   ├── email_alert.py          # HTML email via SMTP
│   ├── whatsapp_alert.py       # WhatsApp via Twilio
│   └── alert_manager.py        # Cooldown + dispatch orchestration
│
├── api/
│   └── server.py               # FastAPI REST backend
│
├── dashboard/
│   └── app.py                  # Streamlit 4-page dashboard
│
├── utils/
│   ├── logger.py               # Rotating file + console logging
│   └── train.py                # YOLOv8 fine-tuning helper
│
├── models/                     # Place your .pt weights here
├── screenshots/                # Auto-saved alert screenshots
└── logs/                       # Application + alert history logs
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/violence-detection.git
cd violence-detection

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

| Variable | Description |
|---|---|
| `MODEL_PATH` | Path to your fine-tuned `.pt` weights |
| `VIDEO_SOURCE` | `0` for webcam, or `rtsp://...` for IP camera |
| `SMTP_USER` / `SMTP_PASSWORD` | Gmail address + App Password |
| `ALERT_RECIPIENTS` | Comma-separated alert email addresses |
| `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` | From Twilio console |
| `TWILIO_WHATSAPP_TO` | Recipient WhatsApp: `whatsapp:+91XXXXXXXXXX` |

### 3. Add Your Model

Place your fine-tuned YOLOv8 weights at the path set in `MODEL_PATH`  
(default: `models/violence_yolov8.pt`).

> If no model file is found, the system falls back to pretrained `yolov8n.pt`  
> which won't detect violence — train your own model first (see below).

---

## 🚀 Running the System

### Option A — Run components separately (development)

```bash
# Terminal 1: Start the FastAPI backend
python -m uvicorn api.server:app --reload --port 8000

# Terminal 2: Start the Streamlit dashboard
streamlit run dashboard/app.py --server.port 8501

# Terminal 3: (Optional) Run pipeline headlessly from CLI
python core/pipeline.py --source 0 --location "Camera-01" --display
```

### Option B — Docker (production)

```bash
docker-compose up --build
```

- Dashboard: http://localhost:8501  
- API docs:  http://localhost:8000/docs

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

### 2. Fine-tune

```bash
python utils/train.py \
  --data    dataset/dataset.yaml \
  --model   yolov8m.pt \
  --epochs  50 \
  --imgsz   640 \
  --batch   16 \
  --device  0 \
  --output  models/violence_yolov8.pt \
  --validate
```

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

### WhatsApp (Twilio Sandbox)

1. Sign up at [twilio.com](https://www.twilio.com)  
2. Go to **Messaging → Try it out → Send a WhatsApp message**  
3. Follow sandbox join instructions (send "join <word>" to sandbox number)  
4. Set credentials in `.env`

For production WhatsApp, request a Twilio WhatsApp Business number.

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/status` | Pipeline stats (FPS, alerts, uptime) |
| GET | `/alerts` | Alert history (paginated) |
| GET | `/alerts/{id}/screenshot` | Fetch screenshot image |
| POST | `/pipeline/start` | Start detection |
| POST | `/pipeline/stop` | Stop detection |
| POST | `/pipeline/config` | Update confidence / cooldown |
| GET | `/stream/frame` | Latest annotated JPEG frame |

Full interactive docs: http://localhost:8000/docs

---

## ⚙️ Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.55` | Minimum detection confidence |
| `FRAME_CONSISTENCY` | `5` | Consecutive violent frames before alert |
| `ALERT_COOLDOWN_SECONDS` | `30` | Min time between alerts |
| `FPS_TARGET` | `20` | Target processing frame rate |
| `MAX_SCREENSHOTS` | `500` | Auto-purge threshold |

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

---

## 📄 License

MIT License — see `LICENSE` for details.
