# Hosted Violence Detection Demo

This directory contains the public portfolio/demo adaptation of the violence detection project.

## Purpose

The production-oriented runtime remains on `fix/pretrained-model-runtime`. This branch adds a zero-cost-hosting-friendly Streamlit entrypoint that calls the existing `core.detector.ViolenceDetector` directly instead of requiring a second FastAPI process.

## Demo features

- Built-in simulated violent sample from the AIRTLab violence-detection dataset
- Built-in non-violent sample from the same dataset
- User video upload (`.mp4`, `.mov`, `.avi`)
- Real YOLOv8 inference using the project's configured pretrained model
- N-frame temporal consistency filtering from the existing detector
- Annotated output video
- Detection timeline and frame-level results
- Alert evidence frames
- CPU-friendly configurable inference sampling
- Public email and WhatsApp delivery intentionally disabled

## Run locally

From the repository root:

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

The app uses the same model configuration as the main project. If the configured model is not present, `core.detector.ViolenceDetector` uses the existing model auto-download path.

## Streamlit Community Cloud

Deploy with:

- Repository: `SahilBh01r1769/violence_detection`
- Branch: `demo/hosted-violence-detection`
- Main file path: `demo/app.py`
- Python: 3.11

`demo/requirements.txt` is intentionally smaller than the production requirements. Root `packages.txt` provides the Linux packages needed by OpenCV and browser-compatible FFmpeg output.

## Architecture

Production-oriented runtime:

```text
CCTV / RTSP / local video
        ↓
FastAPI backend
        ↓
YOLOv8 detector
        ↓
Temporal consistency
        ↓
Alert manager
        ↓
Dashboard / notifications
```

Hosted demo:

```text
Sample / uploaded video
        ↓
Streamlit demo
        ↓
Same core ViolenceDetector
        ↓
YOLOv8 + temporal consistency
        ↓
Annotated result / timeline / demo alert
```

## Attribution

The project integrates third-party pretrained YOLO violence-detection weights. The weights are not presented as original model-training work; see the repository's existing third-party model documentation.

The built-in sample clips are from the AIRTLab **A Dataset for Automatic Violence Detection in Videos**, released for research and educational use:

- Dataset repository: https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos
- M. Bianculli et al., *A dataset for automatic violence detection in videos*, Data in Brief 33 (2020), 106587.
