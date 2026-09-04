import argparse
import csv
from pathlib import Path

import numpy as np
import pytest

from core.detector import Detection, DetectionResult
from evaluation.capture_trace import (
    TimeInterval,
    export_trace,
    is_ground_truth_violent,
    parse_interval,
)


class FakeCapture:
    def __init__(self, frame_count=3, fps=2.0, opened=True):
        self.frames = [np.zeros((2, 2, 3), dtype=np.uint8)] * frame_count
        self.fps = fps
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, _property):
        return self.fps

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class ScriptedDetector:
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def process_frame(self, frame):
        violent, confidence = next(self.decisions)
        detections = []
        if violent:
            detections.append(
                Detection(1, "violence", confidence, (0, 0, 1, 1), True)
            )
        return DetectionResult(
            frame=frame,
            detections=detections,
            is_violent=violent,
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0:2.5", TimeInterval(0.0, 2.5)),
        ("1.25:3", TimeInterval(1.25, 3.0)),
    ],
)
def test_parse_interval(raw, expected):
    assert parse_interval(raw) == expected


@pytest.mark.parametrize("raw", ["2", "-1:2", "2:2", "3:2", "nan:4"])
def test_parse_interval_rejects_invalid_boundaries(raw):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_interval(raw)


def test_ground_truth_intervals_are_half_open():
    intervals = [TimeInterval(1.0, 2.0), TimeInterval(4.0, 5.0)]

    assert is_ground_truth_violent(1.0, intervals)
    assert is_ground_truth_violent(4.5, intervals)
    assert not is_ground_truth_violent(2.0, intervals)
    assert not is_ground_truth_violent(3.0, intervals)


def test_export_trace_records_replayable_frame_decisions(tmp_path: Path):
    capture = FakeCapture(frame_count=3, fps=2.0)
    output = tmp_path / "trace.csv"

    summary = export_trace(
        Path("sample.mp4"),
        output,
        [TimeInterval(0.5, 1.5)],
        detector=ScriptedDetector([(False, 0.9), (True, 0.8), (True, 0.7)]),
        capture_factory=lambda _path: capture,
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["timestamp_seconds"] for row in rows] == [
        "0.000000",
        "0.500000",
        "1.000000",
    ]
    assert [row["is_violent"] for row in rows] == ["False", "True", "True"]
    assert [row["confidence"] for row in rows] == [
        "0.000000",
        "0.800000",
        "0.700000",
    ]
    assert [row["ground_truth_violent"] for row in rows] == [
        "False",
        "True",
        "True",
    ]
    assert summary.frames == 3
    assert summary.positive_frames == 2
    assert summary.source_fps == 2.0
    assert capture.released


def test_export_trace_rejects_invalid_source_fps(tmp_path: Path):
    capture = FakeCapture(fps=0)

    with pytest.raises(RuntimeError, match="invalid frame rate"):
        export_trace(
            Path("sample.mp4"),
            tmp_path / "trace.csv",
            [],
            detector=ScriptedDetector([]),
            capture_factory=lambda _path: capture,
        )

    assert capture.released
