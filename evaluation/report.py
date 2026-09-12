"""Build a compact CSV report from replayable detector traces."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from evaluation.temporal import compare_operating_points, load_trace


def build_rows(traces, confidence_values, positive_values, negative_values):
    rows: list[dict[str, object]] = []
    for clip, path in traces:
        results = compare_operating_points(
            load_trace(path), confidence_values, positive_values, negative_values
        )
        rows.extend({"clip": clip, **asdict(result)} for result in results)
    return rows


def write_csv(rows: Sequence[dict[str, object]], output: Path) -> None:
    if not rows:
        raise ValueError("at least one result row is required")
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
    parser = argparse.ArgumentParser(description="Export a temporal replay CSV report")
    parser.add_argument("--trace", action="append", required=True, type=parse_trace)
    parser.add_argument("--confidence-thresholds", default="0.40,0.55,0.70,0.85")
    parser.add_argument("--thresholds", default="1,3,5,10")
    parser.add_argument("--negative-release-frames", default="1,3")
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows(args.trace, [float(v) for v in args.confidence_thresholds.split(",")], [int(v) for v in args.thresholds.split(",")], [int(v) for v in args.negative_release_frames.split(",")])
    write_csv(rows, args.csv)
    print(f"Wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
