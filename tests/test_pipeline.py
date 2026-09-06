from pathlib import Path

import numpy as np

import alerts.alert_manager as alert_module
from core.detector import Detection, DetectionResult, ViolenceDetector
from core.pipeline import DetectionPipeline, normalise_source


def test_webcam_source_string_becomes_integer():
    assert normalise_source("0") == 0


def test_video_path_stays_string():
    assert normalise_source("sample.mp4") == "sample.mp4"


def test_pipeline_persists_event_when_notifications_are_disabled(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(alert_module, "EVENT_LOG_FILE", tmp_path / "events.json")
    monkeypatch.setattr(ViolenceDetector, "_load_model", lambda self: None)
    pipeline = DetectionPipeline(enable_email=False, enable_whatsapp=False)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    result = DetectionResult(
        frame=frame,
        detections=[Detection(1, "violence", 0.9, (0, 0, 5, 5), True)],
        is_violent=True,
        alert_triggered=True,
    )
    monkeypatch.setattr(pipeline.detector, "process_frame", lambda _frame: result)

    class OneFrameStream:
        end_reason = "ended"

        def __init__(self, source):
            self.source = source

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def frames(self):
            yield frame

        def save_screenshot(self, _frame, prefix):
            assert prefix == "event"
            return Path(tmp_path / "event.jpg")

    monkeypatch.setattr("core.pipeline.VideoStream", OneFrameStream)

    pipeline.run(source="sample.mp4")

    assert pipeline.events_recorded == 1
    assert pipeline.alert_manager.total_events == 1
    event = pipeline.alert_manager.history[0]
    assert event.source == "sample.mp4"
    assert event.notification_status == "not_attempted"
    assert event.notification_suppression_reason == "no_channel_enabled"


def test_notification_bookkeeping_failure_does_not_erase_or_stop_event(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(alert_module, "EVENT_LOG_FILE", tmp_path / "events.json")
    monkeypatch.setattr(ViolenceDetector, "_load_model", lambda self: None)
    pipeline = DetectionPipeline(enable_email=False, enable_whatsapp=False)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    result = DetectionResult(
        frame=frame,
        detections=[Detection(1, "violence", 0.9, (0, 0, 5, 5), True)],
        is_violent=True,
        alert_triggered=True,
    )
    monkeypatch.setattr(pipeline.detector, "process_frame", lambda _frame: result)
    monkeypatch.setattr(
        pipeline.alert_manager,
        "request_notification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("notification state unavailable")
        ),
    )

    class OneFrameStream:
        end_reason = "ended"

        def __init__(self, source):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def frames(self):
            yield frame

        def save_screenshot(self, _frame, prefix):
            return Path(tmp_path / f"{prefix}.jpg")

    monkeypatch.setattr("core.pipeline.VideoStream", OneFrameStream)

    pipeline.run(source="sample.mp4")

    assert pipeline.source_state == "ended"
    assert pipeline.events_recorded == 1
    assert pipeline.alert_manager.total_events == 1
    assert pipeline.last_error.stage == "alert"
    assert pipeline.last_error.message == "notification state unavailable"
