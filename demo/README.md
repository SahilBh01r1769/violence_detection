# VisionGuard Hosted Demo

This directory contains the Streamlit Community Cloud adaptation of the full violence-detection project.

The hosted demo intentionally keeps a smaller architecture than `main`: it analyzes sample or uploaded video directly inside Streamlit while reusing the same `core.detector.ViolenceDetector` and temporal consistency logic.

## What the demo includes

- Built-in violence and normal samples from the AIRTLab violence-detection dataset
- User video upload (`.mp4`, `.mov`, `.avi`)
- Real pretrained YOLOv8 violence/fight inference
- Same N-frame temporal consistency behavior as the full project
- Annotated output video
- Confidence/detection timeline
- Alert evidence frames
- CPU-friendly configurable frame sampling
- Pinned upstream model source for reproducibility
- Fail-closed inference behavior

## What is intentionally omitted

The public hosted demo does **not** expose webcam/RTSP capture, FastAPI controls, persistent production alert history, Email delivery, or WhatsApp delivery. Those remain part of the full project on `main`.

This keeps the Streamlit deployment lightweight while preserving the core inference and temporal-alert behavior.

## Run locally

From the repository root:

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

The model downloads automatically when missing.

## Streamlit Community Cloud

Deploy with:

- Repository: `SahilBh01r1769/violence_detection`
- Branch: `demo/hosted-violence-detection`
- Main file path: `demo/app.py`
- Python: 3.11

`demo/requirements.txt` uses CPU PyTorch, headless OpenCV, and `imageio-ffmpeg` for a Community-Cloud-friendly runtime.

## Architecture

```text
Sample / uploaded video
        ↓
Streamlit hosted demo
        ↓
core.detector.ViolenceDetector
        ↓
Pretrained YOLOv8 inference
        ↓
N-frame temporal consistency
        ↓
Annotated output + timeline + alert evidence
```

Full application architecture and setup instructions are maintained on the [`main`](https://github.com/SahilBh01r1769/violence_detection) branch.

## Attribution

The project integrates third-party pretrained YOLO violence/fight weights. The checkpoint is not presented as original model-training work; model provenance is documented in `THIRD_PARTY_MODELS.md` and on the main branch.

The built-in sample clips are from the AIRTLab **A Dataset for Automatic Violence Detection in Videos**, released for research and educational use:

- Dataset repository: https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos
- M. Bianculli et al., *A dataset for automatic violence detection in videos*, Data in Brief 33 (2020), 106587.
