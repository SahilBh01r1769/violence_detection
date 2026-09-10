"""Compare consecutive qualification with rolling-window voting."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from evaluation.temporal import (
    evaluate_rolling_window,
    evaluate_threshold,
    load_trace,
)


def build_comparison_rows(
    traces,
    confidence: float,
    consecutive_frames: int,
    rolling_minimum: int,
    rolling_window: int,
    negative_release_frames: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for clip, path in traces:
        observations = load_trace(path)
        policies = (
            (
                "consecutive",
                consecutive_frames,
                consecutive_frames,
                evaluate_threshold(
                    observations,
                    consecutive_frames,
                    negative_release_frames,
                    confidence,
                ),
            ),
            (
                "rolling_window",
                rolling_minimum,
                rolling_window,
                evaluate_rolling_window(
                    observations,
                    rolling_minimum,
                    rolling_window,
                    negative_release_frames,
                    confidence,
                ),
            ),
        )
        for strategy, minimum_positives, window_size, metrics in policies:
            values = asdict(metrics)
            values.pop("positive_frames")
            rows.append(
                {
                    "clip": clip,
                    "strategy": strategy,
                    "minimum_positives": minimum_positives,
                    "window_size": window_size,
                    **values,
                }
            )
    return rows


def write_comparison(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        raise ValueError("at least one comparison row is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_trace(value: str):
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("trace must use NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare consecutive and rolling-window temporal policies"
    )
    parser.add_argument("--trace", action="append", required=True, type=parse_trace)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--consecutive-frames", type=int, default=5)
    parser.add_argument("--rolling-minimum", type=int, default=5)
    parser.add_argument("--rolling-window", type=int, default=7)
    parser.add_argument("--negative-release-frames", type=int, default=3)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    rows = build_comparison_rows(
        args.trace,
        args.confidence,
        args.consecutive_frames,
        args.rolling_minimum,
        args.rolling_window,
        args.negative_release_frames,
    )
    write_comparison(rows, args.csv)
    print(f"Wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
