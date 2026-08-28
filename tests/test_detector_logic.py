import numpy as np

from core.detector import Detection, DetectionResult, ViolenceDetector


def test_alert_metadata_prefers_violent_detection():
    result = DetectionResult(frame=np.zeros((10, 10, 3), dtype=np.uint8), detections=[Detection(0, "NoViolence", 0.95, (0, 0, 1, 1), False), Detection(1, "Violence", 0.72, (0, 0, 1, 1), True)], is_violent=True)
    assert result.primary_class == "Violence"
    assert result.max_confidence == 0.72


def test_class_id_one_is_violent_without_loading_model(monkeypatch):
    monkeypatch.setattr(ViolenceDetector, "_load_model", lambda self: None)
    detector = ViolenceDetector("unused.pt", violence_classes=[], violence_class_ids={1})
    assert detector._is_violent_class(1, "anything")
    assert not detector._is_violent_class(0, "NoViolence")


def test_changing_frame_consistency_resets_window(monkeypatch):
    monkeypatch.setattr(ViolenceDetector, "_load_model", lambda self: None)
    detector = ViolenceDetector("unused.pt", frame_consistency=5)
    detector._window.extend([True, True])
    detector.set_frame_consistency(3)
    assert detector.frame_consistency == 3
    assert len(detector._window) == 0
    assert detector._window.maxlen == 3
