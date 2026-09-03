import numpy as np
import pytest

from core.detector import Detection, DetectionResult, ViolenceDetector


class FakeBox:
    def __init__(self, class_id: int, confidence: float = 0.8):
        self.cls = np.array([class_id])
        self.conf = np.array([confidence])
        self.xyxy = np.array([[0, 0, 5, 5]])


class ScriptedModel:
    names = {0: "non_violence", 1: "violence"}

    def __init__(self, decisions):
        self._decisions = iter(decisions)

    def predict(self, **_kwargs):
        is_violent = next(self._decisions)
        boxes = [FakeBox(1)] if is_violent else None
        prediction = type("Prediction", (), {"boxes": boxes})()
        return [prediction]


def detector_for(decisions, frame_consistency):
    detector = ViolenceDetector.__new__(ViolenceDetector)
    detector.confidence = 0.55
    detector.frame_consistency = frame_consistency
    detector.violence_classes = {"violence"}
    detector.violence_class_ids = {1}
    detector._model = ScriptedModel(decisions)
    detector._model_path = None
    detector._frame_id = 0

    from collections import deque

    detector._window = deque(maxlen=frame_consistency)
    return detector


def run_decisions(decisions, frame_consistency):
    detector = detector_for(decisions, frame_consistency)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    return [
        detector.process_frame(frame).alert_triggered
        for _ in decisions
    ]


def test_alert_metadata_prefers_violent_detection():
    result = DetectionResult(
        frame=np.zeros((10, 10, 3), dtype=np.uint8),
        detections=[
            Detection(0, "NoViolence", 0.95, (0, 0, 1, 1), False),
            Detection(1, "Violence", 0.72, (0, 0, 1, 1), True),
        ],
        is_violent=True,
    )
    assert result.primary_class == "Violence"
    assert result.max_confidence == 0.72


def test_class_id_one_is_violent_without_loading_model(monkeypatch):
    monkeypatch.setattr(ViolenceDetector, "_load_model", lambda self: None)
    detector = ViolenceDetector(
        "unused.pt",
        violence_classes=[],
        violence_class_ids={1},
    )
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


@pytest.mark.parametrize(
    ("frame_consistency", "expected"),
    [
        (1, [True]),
        (3, [False, False, True]),
        (5, [False, False, False, False, True]),
        (10, [False] * 9 + [True]),
    ],
)
def test_threshold_triggers_on_nth_consecutive_positive(
    frame_consistency,
    expected,
):
    decisions = [True] * frame_consistency
    assert run_decisions(decisions, frame_consistency) == expected


def test_negative_frame_requires_a_new_positive_run():
    decisions = [True, True, False, True, True, True]
    assert run_decisions(decisions, 3) == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_trigger_clears_window_and_allows_repeat_during_same_positive_run():
    decisions = [True] * 6
    assert run_decisions(decisions, 3) == [
        False,
        False,
        True,
        False,
        False,
        True,
    ]
