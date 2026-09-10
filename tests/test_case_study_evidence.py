from dataclasses import asdict
import csv
import json
from pathlib import Path

import pytest

from evaluation.temporal import compare_operating_points, load_trace
from evaluation.compare_strategies import build_comparison_rows


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


EXPANDED_TRACES = {
    "original_nonviolence": CASE_STUDY / "nonviolence_trace.csv",
    "original_violence": CASE_STUDY / "violence_trace.csv",
    "crowded_nonviolent": CASE_STUDY / "traces/crowded_nonviolent.csv",
    "nonviolent_contact": CASE_STUDY / "traces/nonviolent_contact.csv",
    "nonviolent_walking": CASE_STUDY / "traces/nonviolent_walking.csv",
    "soldiers_with_guns": CASE_STUDY / "traces/soldiers_with_guns.csv",
    "fencing": CASE_STUDY / "traces/fencing.csv",
    "intermittent_violence": CASE_STUDY / "traces/intermittent_violence.csv",
}


def test_expanded_summary_replays_exactly():
    actual = []
    for clip, path in EXPANDED_TRACES.items():
        actual.extend(
            {"clip": clip, **asdict(result)}
            for result in compare_operating_points(
                load_trace(path),
                confidence_thresholds=[0.40, 0.55, 0.70, 0.85],
                positive_frame_values=[1, 3, 5, 10],
                negative_release_values=[1, 3],
            )
        )
    with (CASE_STUDY / "summary.csv").open(newline="", encoding="utf-8") as handle:
        expected = list(csv.DictReader(handle))
    assert len(expected) == 256
    assert [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in actual
    ] == expected


def test_expanded_capture_metadata_is_consistent():
    metadata = [
        json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        for path in list(EXPANDED_TRACES.values())[2:]
    ]
    assert {item["model_sha256"] for item in metadata} == {
        "1714eef088cd4ae5fc8618f6f94890d476434298be6c7a2151faef18bef9d459"
    }
    assert {item["python"] for item in metadata} == {"3.14.5"}
    assert {item["packages"]["ultralytics"] for item in metadata} == {"8.4.142"}


def test_strategy_comparison_replays_exactly():
    actual = build_comparison_rows(
        EXPANDED_TRACES.items(),
        confidence=0.70,
        consecutive_frames=5,
        rolling_minimum=5,
        rolling_window=7,
        negative_release_frames=3,
    )
    with (CASE_STUDY / "strategy_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        expected = list(csv.DictReader(handle))

    assert len(expected) == 16
    assert [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in actual
    ] == expected
