# Temporal Violence Event Filter

A small streaming-systems experiment about one question:

> How should noisy frame-level classifications be converted into an alertable event?

The project uses a third-party YOLOv8 checkpoint as a source of per-frame detections. Its original contribution is the surrounding pipeline: video ingestion, temporal filtering, event creation, delivery-state handling, runtime status, and repeatable comparison of filtering thresholds.

This is not a trained-from-scratch model, a benchmark of model accuracy, or a production safety system.

## Problem

A frame classifier can alternate between positive and negative predictions during the same scene. Triggering on every positive frame produces noisy duplicate alerts. Waiting for several positive frames reduces noise but delays the event.

The experiment compares detector confidence, consecutive-positive thresholds
of 1, 3, 5, and 10 frames, and one- versus three-negative-frame release. It
records:

- false triggers on non-violent footage;
- duplicate triggers inside a ground-truth event;
- video-time delay from ground-truth onset to the alertable event;
- delay introduced by negative-frame release;
- detected and missed ground-truth events.

The repository will report measurements only when they have been produced from identified sample videos. Passing unit tests are not presented as model-accuracy or performance evidence.

## Current pipeline

```text
video source
    -> OpenCV frame reader
    -> pretrained YOLO inference
    -> violent / non-violent frame decision
    -> N-positive / K-negative temporal filter
    -> persisted local event and screenshot
    -> optional Telegram submission and separate outcome
```

The capture and inference path currently runs in one processing loop. The API runs that loop in a background thread so the dashboard remains responsive.

## Scope

The core experiment includes:

- webcam, RTSP, uploaded-video, or local video-path ingestion;
- a pinned third-party YOLOv8 fight/violence checkpoint;
- configurable detection confidence;
- an N-frame consecutive-positive event start and K-frame negative release;
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
- recipient viewing or action after a provider accepts a notification.

YOLO confidence is treated as detector confidence, not as a calibrated probability that a violent event is occurring.

## Evidence status

The committed two-clip case study replays 848 nonviolent observations and 694
violent observations through 32 combinations of confidence, positive-frame,
and negative-release settings. The committed JSON is checked against fresh
replay by the test suite.

The provisional C=0.70, N=5, K=3 setting produced three false events on the
meeting clip and eight triggers, seven of them duplicates, inside the single
continuous violent interval. Its first alert occurred 1.7917 seconds into
that interval in video time. No tested setting both eliminated false events
and detected the violent clip. See
[`evaluation/case_study`](evaluation/case_study/README.md) for the sources,
annotations, complete outputs, reproduction commands, and limitations.

These two clips demonstrate temporal trade-offs but do not estimate general
model accuracy. The measurements do not establish RTSP recovery, end-to-end
wall-clock latency, or a general FPS figure. The test suite separately covers
temporal state transitions, persistence and notification failure paths,
runtime error reporting, API validation, and source-label redaction.

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

The dashboard accepts an uploaded video, a local video-file path, a webcam
index such as `0`, or an RTSP URL.

The headless pipeline can also be run directly:

```bash
python -m core.pipeline --source "path/to/video.mp4" --location "Test-Video"
```

## Detection rule

For each decoded frame:

1. YOLO returns detected boxes, class IDs, labels, and confidence values.
2. A frame is positive when at least one detection matches the configured violence class.
3. N consecutive positive frames start one event and latch it active.
4. Continued positive frames do not emit another event.
5. K consecutive negative frames release the latch so a later positive run
   can become a new event.

N controls evidence required at onset. K separately controls tolerance for
short negative gaps inside an active event. Notification cooldown does not
define event identity and does not suppress local event persistence.

Default configuration:

```text
CONFIDENCE_THRESHOLD=0.70
FRAME_CONSISTENCY=5
NEGATIVE_RELEASE_FRAMES=3
ALERT_COOLDOWN_SECONDS=30
FPS_TARGET=20
```

The C=0.70, N=5, K=3 default is provisional and selected only from the
committed two-clip case study. It must not be described as generally optimal.

`FPS_TARGET` limits ingestion rate. The dashboard's current FPS value is processed frames divided by elapsed runtime; it is not a hardware benchmark.

## Third-party model

The default checkpoint comes from [Musawer1214/Fight-Violence-detection-yolov8](https://github.com/Musawer1214/Fight-Violence-detection-yolov8) and is pinned to upstream commit:

```text
20f0d05054cff7da2dc78dee3c2de1bd54106a13
```

The upstream checkpoint exposes `non_violence` and `violence`, with class ID 1 treated as violent by default. See [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md).

## Repeatable filter comparison

Each sample video is inferred once at a 0.25 capture floor into a saved
frame-level trace. Confidence thresholds of 0.40, 0.55, 0.70, and 0.85;
positive thresholds of 1, 3, 5, and 10; and negative-release values of 1 and
3 then replay the same detector outputs. Differences therefore come from the
decision settings rather than separate inference runs.

```bash
python -m evaluation.temporal evaluation/case_study/violence_trace.csv --confidence-thresholds 0.40,0.55,0.70,0.85 --thresholds 1,3,5,10 --negative-release-frames 1,3
```

## License and responsible use

Use only footage and camera sources you are authorized to process. This experiment is unsuitable for autonomous safety decisions or unreviewed surveillance.
