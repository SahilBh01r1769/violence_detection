"""Replay saved frame decisions through consecutive-frame filters.

The evaluator intentionally does not load YOLO. A sample video is inferred once,
then its CSV trace can be replayed cheaply for several temporal thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence


@dataclass(frozen=True)
class FrameObservation:
    frame_id: int
    timestamp_seconds: float
    is_violent: bool
    confidence: float
    ground_truth_violent: bool


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: int
    total_triggers: int
    false_triggers: int
    duplicate_triggers: int
    detected_events: int
    missed_events: int
    mean_alert_delay_seconds: float | None


def _as_bool(value: str) -> bool:
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes"}:
        return True
    if normalised in {"0", "false", "no"}:
        return False
    raise ValueError(f"Expected a Boolean value, received {value!r}")


def load_trace(path: Path) -> list[FrameObservation]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "frame_id",
            "timestamp_seconds",
            "is_violent",
            "confidence",
            "ground_truth_violent",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Trace is missing columns: {', '.join(sorted(missing))}")
        return [
            FrameObservation(
                frame_id=int(row["frame_id"]),
                timestamp_seconds=float(row["timestamp_seconds"]),
                is_violent=_as_bool(row["is_violent"]),
                confidence=float(row["confidence"]),
                ground_truth_violent=_as_bool(row["ground_truth_violent"]),
            )
            for row in reader
        ]


def trigger_indices(
    observations: Sequence[FrameObservation], threshold: int
) -> list[int]:
    if threshold < 1:
        raise ValueError("threshold must be at least 1")

    positive_run = 0
    event_active = False
    triggers: list[int] = []
    for index, observation in enumerate(observations):
        if not observation.is_violent:
            positive_run = 0
            event_active = False
            continue

        positive_run += 1
        if not event_active and positive_run >= threshold:
            triggers.append(index)
            event_active = True
    return triggers


def ground_truth_events(
    observations: Sequence[FrameObservation],
) -> list[tuple[int, int]]:
    events: list[tuple[int, int]] = []
    start: int | None = None
    for index, observation in enumerate(observations):
        if observation.ground_truth_violent and start is None:
            start = index
        if start is not None and not observation.ground_truth_violent:
            events.append((start, index - 1))
            start = None
    if start is not None:
        events.append((start, len(observations) - 1))
    return events


def evaluate_threshold(
    observations: Sequence[FrameObservation], threshold: int
) -> ThresholdMetrics:
    triggers = trigger_indices(observations, threshold)
    events = ground_truth_events(observations)
    false_triggers = sum(
        not observations[index].ground_truth_violent for index in triggers
    )

    detected_events = 0
    duplicate_triggers = 0
    delays: list[float] = []
    for start, end in events:
        event_triggers = [index for index in triggers if start <= index <= end]
        if not event_triggers:
            continue
        detected_events += 1
        duplicate_triggers += max(0, len(event_triggers) - 1)
        delays.append(
            observations[event_triggers[0]].timestamp_seconds
            - observations[start].timestamp_seconds
        )

    return ThresholdMetrics(
        threshold=threshold,
        total_triggers=len(triggers),
        false_triggers=false_triggers,
        duplicate_triggers=duplicate_triggers,
        detected_events=detected_events,
        missed_events=len(events) - detected_events,
        mean_alert_delay_seconds=round(mean(delays), 4) if delays else None,
    )


def compare_thresholds(
    observations: Sequence[FrameObservation], thresholds: Iterable[int]
) -> list[ThresholdMetrics]:
    return [evaluate_threshold(observations, value) for value in thresholds]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare temporal thresholds using a saved frame trace"
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--thresholds", default="1,3,5,10")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    thresholds = [int(value) for value in args.thresholds.split(",")]
    results = compare_thresholds(load_trace(args.trace), thresholds)
    rendered = json.dumps([asdict(result) for result in results], indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
