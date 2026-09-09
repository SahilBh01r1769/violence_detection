import time

import numpy as np
import pytest

from core.detector import ViolenceDetector
from core.pipeline import DetectionPipeline
from core.stream import VideoStream, safe_source_label


def test_rtsp_password_is_redacted():
    value = safe_source_label("rtsp://user:secret@example.com:554/stream")
    assert "secret" not in value
    assert "***" in value


def test_inference_failure_is_not_treated_as_safe(monkeypatch):
    monkeypatch.setattr(ViolenceDetector, "_load_model", lambda self: None)
    detector = ViolenceDetector("unused.pt")
    class BrokenModel:
        names = {0: "non_violence", 1: "violence"}
        def predict(self, **kwargs):
            raise ValueError("boom")
    detector._model = BrokenModel()
    with pytest.raises(RuntimeError):
        detector.process_frame(np.zeros((10, 10, 3), dtype=np.uint8))


def test_stopped_uptime_is_frozen(monkeypatch):
    monkeypatch.setattr(ViolenceDetector, "_load_model", lambda self: None)
    pipeline = DetectionPipeline()
    pipeline.start_time = time.time() - 2
    pipeline.end_time = time.time()
    pipeline._running = False
    first = pipeline.uptime
    time.sleep(0.02)
    second = pipeline.uptime
    assert second == pytest.approx(first, abs=0.01)


def pipeline_without_model(monkeypatch):
    monkeypatch.setattr(ViolenceDetector, "_load_model", lambda self: None)
    return DetectionPipeline()


def test_source_open_failure_is_retained(monkeypatch):
    pipeline = pipeline_without_model(monkeypatch)

    class BrokenStream:
        def __init__(self, source):
            pass

        def __enter__(self):
            raise RuntimeError("camera unavailable")

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr("core.pipeline.VideoStream", BrokenStream)

    pipeline.run(source=0)

    assert pipeline.source_state == "error"
    assert pipeline.last_error.stage == "source"
    assert pipeline.last_error.message == "camera unavailable"


def test_inference_failure_is_identified_by_stage(monkeypatch):
    pipeline = pipeline_without_model(monkeypatch)

    class OneFrameStream:
        end_reason = None

        def __init__(self, source):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def frames(self):
            yield np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setattr("core.pipeline.VideoStream", OneFrameStream)
    monkeypatch.setattr(
        pipeline.detector,
        "process_frame",
        lambda _frame: (_ for _ in ()).throw(RuntimeError("inference failed")),
    )

    pipeline.run(source="sample.mp4")

    assert pipeline.source_state == "error"
    assert pipeline.last_error.stage == "inference"
    assert pipeline.last_error.message == "inference failed"


def test_unrecoverable_stream_read_is_reported_as_disconnected(monkeypatch):
    pipeline = pipeline_without_model(monkeypatch)

    class DisconnectedStream:
        end_reason = "disconnected"

        def __init__(self, source):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def frames(self):
            return iter(())

    monkeypatch.setattr("core.pipeline.VideoStream", DisconnectedStream)

    pipeline.run(source="rtsp://example.test/stream")

    assert pipeline.source_state == "disconnected"
    assert pipeline.last_error.stage == "source"
    assert pipeline.last_error.message == "video source disconnected"


def test_stream_records_failed_reconnect(monkeypatch, tmp_path):
    class UnreadableCapture:
        def __init__(self):
            self.opened = True

        def isOpened(self):
            return self.opened

        def read(self):
            return False, None

        def release(self):
            self.opened = False

    monkeypatch.setattr("core.stream.cv2.VideoCapture", lambda _source: UnreadableCapture())
    monkeypatch.setattr("core.stream.time.sleep", lambda _seconds: None)
    stream = VideoStream("rtsp://example.test/stream", screenshot_dir=tmp_path)
    stream._cap = UnreadableCapture()

    assert list(stream.frames()) == []
    assert stream.end_reason == "disconnected"
