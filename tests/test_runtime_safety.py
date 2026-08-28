import time

import numpy as np
import pytest

from core.detector import ViolenceDetector
from core.pipeline import DetectionPipeline
from core.stream import safe_source_label


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
