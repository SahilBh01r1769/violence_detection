# Temporal Violence Event Filter

A small streaming-systems experiment about one question:

> How should noisy frame-level classifications be converted into an alertable event?

The project uses a third-party YOLOv8 checkpoint as a source of per-frame detections. Its original contribution is the surrounding pipeline: video ingestion, temporal filtering, event creation, delivery-state handling, runtime status, and repeatable comparison of filtering thresholds.

This is not a trained-from-scratch model, a benchmark of model accuracy, or a production safety system.

## Problem

A frame classifier can alternate between positive and negative predictions during the same scene. Triggering on every positive frame produces noisy duplicate alerts. Waiting for several positive frames reduces noise but delays the event.

The experiment compares consecutive-positive thresholds of 1, 3, 5, and 10 frames and records:

- false triggers on non-violent footage;
- delay from the first positive frame to the alertable event;
- processed-frame throughput;
- source and inference failures;
- the final state of each locally recorded event.

The repository will report measurements only when they have been produced from identified sample videos. Passing unit tests are not presented as model-accuracy or performance evidence.

## Current pipeline

```text
video source
    -> OpenCV frame reader
    -> pretrained YOLO inference
    -> violent / non-violent frame decision
    -> consecutive-frame filter
    -> local alert event
    -> persisted outcome and screenshot
```

The capture and inference path currently runs in one processing loop. The API runs that loop in a background thread so the dashboard remains responsive.

## Scope

The core experiment includes:

- webcam, RTSP, or local video-path ingestion;
- a pinned third-party YOLOv8 fight/violence checkpoint;
- configurable detection confidence;
- an N-frame consecutive-positive filter;
- local event history and screenshots;
- FastAPI status and control endpoints;
- a small Streamlit experiment view;
- unit tests and replayable temporal-filter evaluation.

## Non-goals

The project does not claim:

- ownership or training of the supplied checkpoint;
- calibrated violence probabilities;
- validated model accuracy;
- guaranteed real-time throughput on every machine;
- reliable safety or surveillance use;
- cloud-scale or production-ready deployment;
- verified external notification delivery.

YOLO confidence is treated as detector confidence, not as a calibrated probability that a violent event is occurring.

## Evidence status

The following are currently established by repository tests:

- configured class ID/name matching;
- preference for violent-detection metadata;
- runtime filter reconfiguration;
- API input validation and configuration updates;
- source normalization;
- inference exceptions are raised rather than converted into safe frames;
- RTSP credentials are redacted in log labels.

The tests do not yet establish temporal trade-offs, notification delivery, RTSP recovery, Docker behavior, end-to-end video accuracy, or a specific FPS figure.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
python -m utils.download_model
```

Start the API:

```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Start the dashboard in a second terminal:

```bash
streamlit run dashboard/app.py --server.port 8501
```

The dashboard accepts a webcam index such as `0`, an RTSP URL, or a local video-file path. It does not upload a video file to the server.

The headless pipeline can also be run directly:

```bash
python -m core.pipeline --source "path/to/video.mp4" --location "Test-Video"
```

## Detection rule

For each decoded frame:

1. YOLO returns detected boxes, class IDs, labels, and confidence values.
2. A frame is positive when at least one detection matches the configured violence class.
3. The Boolean result enters a fixed-length window.
4. An event becomes eligible when all N positions in the window are positive.
5. After a trigger, the window is cleared.

With a fixed N, this is functionally an N-consecutive-positive rule: a negative result must leave the window before a trigger can occur.

Default configuration:

```text
CONFIDENCE_THRESHOLD=0.55
FRAME_CONSISTENCY=5
ALERT_COOLDOWN_SECONDS=30
FPS_TARGET=20
```

`FPS_TARGET` limits ingestion rate. The dashboard's current FPS value is processed frames divided by elapsed runtime; it is not a hardware benchmark.

## Third-party model

The default checkpoint comes from [Musawer1214/Fight-Violence-detection-yolov8](https://github.com/Musawer1214/Fight-Violence-detection-yolov8) and is pinned to upstream commit:

```text
20f0d05054cff7da2dc78dee3c2de1bd54106a13
```

The upstream checkpoint exposes `non_violence` and `violence`, with class ID 1 treated as violent by default. See [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md).

## Hardware note

Development targets CPU-only machines as well as GPU-equipped systems. Video inference may run below the video's native frame rate on low-power hardware. Repeatable filter comparisons will therefore operate on saved per-frame traces after a single inference pass, avoiding repeated model execution.

## License and responsible use

Use only footage and camera sources you are authorized to process. This experiment is unsuitable for autonomous safety decisions or unreviewed surveillance.
