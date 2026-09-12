"""Reproduce the frozen-policy evaluation on the held-out clip traces."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

from evaluation.compare_strategies import build_comparison_rows, write_comparison
from evaluation.temporal import ground_truth_events, load_trace


DEFAULT_ROOT = Path(__file__).resolve().parent / "held_out"


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"clip", "reviewed_label", "source_url"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    names = [row["clip"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Manifest clip names must be unique")
    return rows


def build_held_out_rows(
    manifest_path: Path, trace_dir: Path
) -> list[dict[str, object]]:
    manifest = load_manifest(manifest_path)
    traces = []
    for item in manifest:
        path = trace_dir / f"{item['clip']}.csv"
        observations = load_trace(path)
        has_event = bool(ground_truth_events(observations))
        expected_event = item["reviewed_label"] == "violence"
        if has_event != expected_event:
            raise ValueError(
                f"Manifest and trace annotation disagree for {item['clip']}"
            )
        traces.append((item["clip"], path))
    return build_comparison_rows(
        traces,
        confidence=0.70,
        consecutive_frames=5,
        rolling_minimum=5,
        rolling_window=7,
        negative_release_frames=3,
    )


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregates = []
    for strategy in ("consecutive", "rolling_window"):
        selected = [row for row in rows if row["strategy"] == strategy]
        delays = [
            float(row["mean_alert_delay_seconds"])
            for row in selected
            if row["mean_alert_delay_seconds"] is not None
        ]
        aggregates.append(
            {
                "strategy": strategy,
                "clips": len(selected),
                "total_triggers": sum(int(row["total_triggers"]) for row in selected),
                "false_triggers": sum(int(row["false_triggers"]) for row in selected),
                "duplicate_triggers": sum(
                    int(row["duplicate_triggers"]) for row in selected
                ),
                "detected_events": sum(int(row["detected_events"]) for row in selected),
                "missed_events": sum(int(row["missed_events"]) for row in selected),
                "mean_alert_delay_seconds": round(mean(delays), 4) if delays else None,
            }
        )
    return aggregates


def write_rows(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        raise ValueError("at least one row is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the frozen temporal-policy held-out evaluation"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "manifest.csv")
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_ROOT / "traces")
    parser.add_argument(
        "--comparison-csv", type=Path, default=DEFAULT_ROOT / "strategy_comparison.csv"
    )
    parser.add_argument(
        "--aggregate-csv", type=Path, default=DEFAULT_ROOT / "aggregate.csv"
    )
    args = parser.parse_args()

    comparison = build_held_out_rows(args.manifest, args.trace_dir)
    write_comparison(comparison, args.comparison_csv)
    write_rows(aggregate_rows(comparison), args.aggregate_csv)
    print(f"Wrote {len(comparison)} per-clip rows and 2 aggregate rows")


if __name__ == "__main__":
    main()
