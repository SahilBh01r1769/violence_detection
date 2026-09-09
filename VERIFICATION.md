# Verification record

This document records the evidence behind the current Violence Detection project claims. It is a reproducibility aid, not a claim of production safety or general model accuracy.

## Verification commands

From the repository root:

```powershell
python -m pytest -q
python -m evaluation.temporal evaluation/case_study/nonviolence_trace.csv --confidence-thresholds 0.40,0.55,0.70,0.85 --thresholds 1,3,5,10 --negative-release-frames 1,3
python -m evaluation.temporal evaluation/case_study/violence_trace.csv --confidence-thresholds 0.40,0.55,0.70,0.85 --thresholds 1,3,5,10 --negative-release-frames 1,3
```

The committed evidence test regenerates both result matrices from the committed CSV traces and fails if any value changes.

## Claim-to-evidence map

| Claim | Evidence | Evidence type | Limit |
|---|---|---|---|
| Temporal state is shared by live and replay paths | Runtime/replay tests and `evaluation/temporal.py` | Automated tests | Does not measure wall-clock throughput |
| Events persist independently of notification policy | `tests/test_pipeline.py`, `tests/test_alert_manager.py` | Mocked failure and suppression tests | Local filesystem only |
| Telegram is the sole optional external channel | Alert-manager and Telegram sender tests | Mocked HTTP tests | Provider reliability is unverified |
| Telegram image submission works with configured credentials | User reported receiving event images on 2026-09-08 | External manual verification | Does not prove recipient viewing or general delivery reliability |
| `accepted` is provider acceptance | Telegram response handling and persisted status fields | Code and tests | No delivery/read receipt exists |
| Source and runtime failures remain visible | API and pipeline tests | Automated tests | RTSP recovery and hardware failure paths remain unverified |
| Upload, local path, webcam, and RTSP inputs remain supported | Dashboard input tests and source implementation | Automated tests | Webcam/RTSP were not exercised in this environment |

## Sample case study

The two traces under `evaluation/case_study/` were produced from the Mixkit clips documented in that directory. The meeting clip is annotated nonviolent throughout; the martial-arts clip is annotated violent throughout. The traces contain 848 and 694 observations respectively and preserve the detector outputs used for every comparison.

The provisional operating point is confidence `0.70`, five consecutive positive frames, and three consecutive negative frames for release. On these two clips it reduced false events and duplicate triggers relative to the earlier policy, but it did not eliminate false positives. A two-clip case study cannot establish general accuracy, safety, or whether `K=3` merges separate nearby events.

## Final limitations

- The detector is a pretrained, frame-based YOLO model; no temporal model was trained.
- Frame thresholds describe video-time qualification delay, not end-to-end inference and notification latency.
- No broad benchmark, FPS claim, or recipient-level notification claim is made.
- Local history is intentionally simple; it is not a distributed queue or database.
- Telegram credentials belong in a local `.env` file and are never committed.
