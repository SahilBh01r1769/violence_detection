import csv

from evaluation.compare_strategies import build_comparison_rows, write_comparison


def test_comparison_uses_identical_trace_for_both_policies(tmp_path):
    trace = tmp_path / "clip.csv"
    trace.write_text(
        "frame_id,timestamp_seconds,is_violent,confidence,ground_truth_violent,capture_confidence_floor\n"
        "1,0.0,True,0.8,True,0.25\n"
        "2,0.1,True,0.8,True,0.25\n"
        "3,0.2,False,0.0,True,0.25\n"
        "4,0.3,True,0.8,True,0.25\n",
        encoding="utf-8",
    )

    rows = build_comparison_rows(
        [("clip", trace)],
        confidence=0.7,
        consecutive_frames=3,
        rolling_minimum=3,
        rolling_window=4,
        negative_release_frames=2,
    )

    assert len(rows) == 2
    assert rows[0]["strategy"] == "consecutive"
    assert rows[0]["missed_events"] == 1
    assert rows[1]["strategy"] == "rolling_window"
    assert rows[1]["detected_events"] == 1
    assert rows[1]["mean_alert_delay_seconds"] == 0.3

    output = tmp_path / "comparison.csv"
    write_comparison(rows, output)
    with output.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
