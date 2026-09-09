"""Build a compact CSV and SVG report from replayable detector traces."""

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


def write_svg(rows: Sequence[dict[str, object]], output: Path) -> None:
    points = [row for row in rows if row["mean_alert_delay_seconds"] is not None]
    if not points:
        raise ValueError("the report has no detected events to plot")
    width, height, margin = 900, 520, 70
    max_x = max(float(row["false_triggers"]) for row in points) or 1.0
    max_y = max(float(row["mean_alert_delay_seconds"]) for row in points) or 1.0
    circles = []
    for row in points:
        cx = margin + float(row["false_triggers"]) / max_x * (width - 2 * margin)
        cy = height - margin - float(row["mean_alert_delay_seconds"]) / max_y * (height - 2 * margin)
        radius = 4 + min(12, int(row["duplicate_triggers"]))
        label = f'{row["clip"]}: C={row["confidence_threshold"]}, N={row["positive_frames"]}, K={row["negative_release_frames"]}'
        circles.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="#2563eb" fill-opacity="0.45"><title>{label}</title></circle>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="450" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">Temporal-filter trade-off</text>
<line x1="70" y1="450" x2="830" y2="450" stroke="#111"/><line x1="70" y1="70" x2="70" y2="450" stroke="#111"/>
<text x="450" y="502" text-anchor="middle" font-family="sans-serif">False triggers</text>
<text x="18" y="260" text-anchor="middle" transform="rotate(-90 18 260)" font-family="sans-serif">First-alert delay (video seconds)</text>
{''.join(circles)}<text x="830" y="480" text-anchor="end" font-family="sans-serif" font-size="12">Circle size indicates duplicate triggers</text></svg>
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")


def parse_trace(value: str):
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("trace must use NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a temporal replay report")
    parser.add_argument("--trace", action="append", required=True, type=parse_trace)
    parser.add_argument("--confidence-thresholds", default="0.40,0.55,0.70,0.85")
    parser.add_argument("--thresholds", default="1,3,5,10")
    parser.add_argument("--negative-release-frames", default="1,3")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows(args.trace, [float(v) for v in args.confidence_thresholds.split(",")], [int(v) for v in args.thresholds.split(",")], [int(v) for v in args.negative_release_frames.split(",")])
    write_csv(rows, args.csv)
    write_svg(rows, args.svg)
    print(f"Wrote {len(rows)} rows to {args.csv} and {args.svg}")


if __name__ == "__main__":
    main()
