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

The capture and inference path runs in one processing loop. The API runs that
loop in a background thread. Telegram submission uses a separate worker.
Background execution does not guarantee dashboard responsiveness under load.

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

Copy `.env.example` to `.env` in the repository root, beside `config.py`.
Existing environment variables and dashboard settings may override defaults.
Restart the API and dashboard after changing `.env`.

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

## Local events and optional Telegram

Every qualified event is recorded before notification policy is considered.
`logs/event_history.json` stores the event ID, detection timestamp, source,
class, confidence, screenshot path, and separate notification fields.
Screenshots are written under `screenshots/`. History writes use temporary
files followed by replacement. The persisted history retains the latest 1,000
records. Screenshot retention defaults to 500 images, so older records may
outlive their images. This is bounded local history, not an archival store.

Telegram is disabled by default. To enable it, set these in the root `.env`:

```dotenv
ENABLE_TELEGRAM_ALERTS=true
TELEGRAM_BOT_TOKEN=your_private_bot_token
TELEGRAM_CHAT_ID=your_recipient_chat_id
```

Create the bot through Telegram's BotFather and send `/start` to it from the
intended recipient account. Use the numeric ID of that conversation, not the
bot's own ID. Keep the token outside Git and logs. In the dashboard, enable
Telegram in Settings and save before starting a run. Telegram receives the
image directly through an HTTP photo upload; no hosted-media service is needed.

| Notification state | Meaning |
| --- | --- |
| `not_attempted` | Disabled, unconfigured, another submission pending, or cooldown active; the suppression reason identifies which |
| `queued` | Submission worker scheduled; not a successful message |
| `accepted` | Telegram API accepted the submission; no claim about recipient viewing |
| `failed` | Submission failed, or its outcome became unknown after restart |

Cooldown starts only after acceptance. Events occurring during cooldown or
while another submission is pending still persist locally. Failure releases
the pending guard, but does not automatically retry the same latched event.
There is no durable delivery queue. A process exit may interrupt a worker;
previously queued records are treated as failed with an unknown-outcome error
when loaded again. Legacy sender-success history remains readable as accepted
submission, never as proof of recipient viewing.

The project owner reported receiving Telegram images during a real test on
September 8, 2026. Earlier exported records also demonstrated actual provider
rejection of a bot-as-recipient configuration. These observations establish
one working setup and one rejection path, not general delivery reliability.

## Runtime and dashboard

The dashboard provides source selection, start/stop, the current frame,
temporal settings, active-event state, event history, screenshots, notification
outcomes, and the last runtime error. Refresh is optional and disabled by
default. With refresh disabled, displayed status changes on the next rerun.
Uploaded videos are stored in `runtime_uploads/`; the dashboard and API must
share that filesystem. Uploaded files are not automatically purged.

`/status` retains diagnostics while idle. Normal file completion is `ended`,
manual stopping is `stopped`, and an unrecovered live-source read is
`disconnected`. Source opening, inference, and pipeline exceptions retain an
error stage, message, and UTC timestamp. Notification rejection is recorded
on the event and does not erase it. The live reader makes one reconnect
attempt; real RTSP recovery has not been validated. An unsuccessful file read
is treated as completion, so this does not prove a corrupt file decoded fully.

Run one pipeline process against a history directory. Local JSON persistence
has no coordination for multiple independent writers. Stop is cooperative;
an in-progress OpenCV read or model call can delay shutdown.

## Verification and scope

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover state transitions, shared runtime/replay logic, API behavior,
failure visibility, persistence independent of notification eligibility,
mocked Telegram acceptance/rejection, upload storage, and exact replay of
the two saved traces. Synthetic and mocked checks establish implementation
behavior. Saved real detector traces establish only this case study.

The operating point was selected on the same two clips being reported, with
no independent holdout. It is a descriptive selection, not evidence of
generalization. Ground truth is the owner's whole-clip annotation; it has not
been independently reviewed. The capture artifacts do not include the local
checkpoint hash or exact dependency versions, so bit-for-bit re-inference
cannot be guaranteed. Replay of the committed scores is reproducible.

The model remains pretrained and frame-based. No new model was trained.
Email, Twilio/WhatsApp, generic dashboard analytics, the unused training
helper, and unverified Docker deployment files have been removed. Plotly and
direct pandas usage were removed with analytics; Streamlit may still install
pandas transitively. The supported workflow is local API plus dashboard or
the headless pipeline. Real-time latency, general accuracy, RTSP recovery,
and sustained unattended operation remain unverified.

## License and responsible use

Use only footage and camera sources you are authorized to process. This experiment is unsuitable for autonomous safety decisions or unreviewed surveillance.
