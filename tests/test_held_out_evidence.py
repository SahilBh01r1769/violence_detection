import csv
import hashlib
import json
from pathlib import Path

from evaluation.held_out_report import (
    aggregate_rows,
    build_held_out_rows,
    load_manifest,
)
from evaluation.temporal import load_trace


ROOT = Path(__file__).resolve().parents[1] / "evaluation" / "held_out"


def _normalise(rows):
    return [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in rows
    ]


def test_held_out_manifest_and_traces_are_complete():
    manifest = load_manifest(ROOT / "manifest.csv")
    assert len(manifest) == 27
    assert sum(row["reviewed_label"] == "violence" for row in manifest) == 9
    assert sum(row["reviewed_label"] == "non_violence" for row in manifest) == 18

    for row in manifest:
        trace = ROOT / "traces" / f"{row['clip']}.csv"
        metadata = json.loads(trace.with_suffix(".metadata.json").read_text())
        assert trace.exists()
        assert load_trace(trace)
        assert metadata["trace"] == trace.name
        assert len(metadata["source_video_sha256"]) == 64
        assert metadata["model_sha256"] == (
            "1714eef088cd4ae5fc8618f6f94890d476434298be6c7a2151faef18bef9d459"
        )


def test_held_out_results_replay_exactly():
    actual = build_held_out_rows(ROOT / "manifest.csv", ROOT / "traces")
    with (ROOT / "strategy_comparison.csv").open(newline="", encoding="utf-8") as handle:
        expected = list(csv.DictReader(handle))
    assert len(expected) == 54
    assert _normalise(actual) == expected

    with (ROOT / "aggregate.csv").open(newline="", encoding="utf-8") as handle:
        expected_aggregate = list(csv.DictReader(handle))
    assert _normalise(aggregate_rows(actual)) == expected_aggregate
