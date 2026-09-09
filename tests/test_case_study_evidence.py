from dataclasses import asdict
import json
from pathlib import Path

import pytest

from evaluation.temporal import compare_operating_points, load_trace


CASE_STUDY = Path(__file__).resolve().parents[1] / "evaluation" / "case_study"


@pytest.mark.parametrize(
    ("name", "expected_frames", "expected_last_timestamp", "truth_value"),
    [
        ("nonviolence", 848, 28.233333, False),
        ("violence", 694, 28.875, True),
    ],
)
def test_case_study_results_replay_exactly(
    name,
    expected_frames,
    expected_last_timestamp,
    truth_value,
):
    observations = load_trace(CASE_STUDY / f"{name}_trace.csv")

    assert len(observations) == expected_frames
    assert observations[-1].timestamp_seconds == expected_last_timestamp
    assert {item.ground_truth_violent for item in observations} == {truth_value}
    assert {item.capture_confidence_floor for item in observations} == {0.25}

    actual = [
        asdict(result)
        for result in compare_operating_points(
            observations,
            confidence_thresholds=[0.40, 0.55, 0.70, 0.85],
            positive_frame_values=[1, 3, 5, 10],
            negative_release_values=[1, 3],
        )
    ]
    expected = json.loads(
        (CASE_STUDY / f"{name}_results.json").read_text(encoding="utf-8")
    )

    assert actual == expected
