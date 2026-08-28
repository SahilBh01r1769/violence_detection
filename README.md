# VisionGuard — Hosted Violence Detection Demo

This branch powers the public Streamlit demo for the full [`violence_detection`](https://github.com/SahilBh01r1769/violence_detection) project.

It intentionally keeps a lightweight hosted architecture while reusing the same pretrained YOLOv8 violence detector and N-frame temporal consistency logic as the main project.

## Demo flow

```text
Sample / uploaded video
        ↓
Streamlit hosted demo
        ↓
Same core ViolenceDetector
        ↓
Pretrained YOLOv8 inference
        ↓
N-frame temporal consistency
        ↓
Annotated output + timeline + alert evidence
```

## Included in this hosted branch

- Built-in violence and normal sample clips
- User video uploads (`.mp4`, `.mov`, `.avi`)
- Real YOLOv8 inference
- Temporal alert filtering
- Annotated output video
- Confidence/detection timeline
- Alert evidence frames
- CPU-friendly sampled inference
- Streamlit Community Cloud configuration

## Intentionally omitted from the public demo

The complete `main` project additionally supports webcam/RTSP sources, FastAPI runtime controls, persistent alert history, Email alerts, WhatsApp alerts, Docker Compose, and the full monitoring dashboard.

Those features are intentionally not exposed through this lightweight public deployment.

## Run the demo locally

```bash
python -m venv venv
```

Activate the environment, then:

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

The default pretrained model downloads automatically when missing.

## Streamlit Community Cloud

Use:

```text
Repository:     SahilBh01r1769/violence_detection
Branch:         demo/hosted-violence-detection
Main file path: demo/app.py
Python:         3.11
```

## Full project

For the production-oriented architecture, complete setup guide, FastAPI service, dashboard, Docker configuration, notification integrations, tests, and model provenance, use the [`main` branch](https://github.com/SahilBh01r1769/violence_detection).

## Attribution

The pretrained YOLO violence/fight checkpoint is third-party and is not presented as original model-training work. The hosted branch pins the upstream model source for reproducibility.

The built-in sample clips are from the AIRTLab *A Dataset for Automatic Violence Detection in Videos* and are used as research/educational demo samples.

See [`demo/README.md`](demo/README.md) and [`THIRD_PARTY_MODELS.md`](THIRD_PARTY_MODELS.md) for details.
