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

from core.temporal import TemporalEventFilter


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
    negative_release_frames: int
    total_triggers: int
    false_triggers: int
    duplicate_triggers: int
    detected_events: int
    missed_events: int
    mean_alert_delay_seconds: float | None
    mean_release_delay_seconds: float | None
    merged_ground_truth_events: int


@dataclass(frozen=True)
class TemporalReplay:
    trigger_indices: tuple[int, ...]
    active_intervals: tuple[tuple[int, int], ...]
    release_delays_seconds: tuple[float, ...]


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


def replay_temporal_filter(
    observations: Sequence[FrameObservation],
    threshold: int,
    negative_release_frames: int = 1,
) -> TemporalReplay:
    temporal_filter = TemporalEventFilter(threshold, negative_release_frames)
    triggers: list[int] = []
    active_intervals: list[tuple[int, int]] = []
    release_delays: list[float] = []
    event_start: int | None = None
    first_negative_timestamp: float | None = None

    for index, observation in enumerate(observations):
        was_active = temporal_filter.event_active
        if was_active and not observation.is_violent:
            first_negative_timestamp = (
                observation.timestamp_seconds
                if first_negative_timestamp is None
                else first_negative_timestamp
            )

        decision = temporal_filter.update(observation.is_violent)
        if decision.triggered:
            triggers.append(index)
            event_start = index
        if decision.event_active and observation.is_violent:
            first_negative_timestamp = None
        if decision.released and event_start is not None:
            active_intervals.append((event_start, index - 1))
            if first_negative_timestamp is not None:
                release_delays.append(
                    observation.timestamp_seconds - first_negative_timestamp
                )
            event_start = None
            first_negative_timestamp = None

    if temporal_filter.event_active and event_start is not None:
        active_intervals.append((event_start, len(observations) - 1))

    return TemporalReplay(
        tuple(triggers),
        tuple(active_intervals),
        tuple(release_delays),
    )


def trigger_indices(
    observations: Sequence[FrameObservation],
    threshold: int,
    negative_release_frames: int = 1,
) -> list[int]:
    return list(
        replay_temporal_filter(
            observations,
            threshold,
            negative_release_frames,
        ).trigger_indices
    )


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
    observations: Sequence[FrameObservation],
    threshold: int,
    negative_release_frames: int = 1,
) -> ThresholdMetrics:
    replay = replay_temporal_filter(
        observations,
        threshold,
        negative_release_frames,
    )
    triggers = replay.trigger_indices
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

    merged_ground_truth_events = sum(
        max(
            0,
            sum(
                active_start <= truth_end and truth_start <= active_end
                for truth_start, truth_end in events
            )
            - 1,
        )
        for active_start, active_end in replay.active_intervals
    )

    return ThresholdMetrics(
        threshold=threshold,
        negative_release_frames=negative_release_frames,
        total_triggers=len(triggers),
        false_triggers=false_triggers,
        duplicate_triggers=duplicate_triggers,
        detected_events=detected_events,
        missed_events=len(events) - detected_events,
        mean_alert_delay_seconds=round(mean(delays), 4) if delays else None,
        mean_release_delay_seconds=(
            round(mean(replay.release_delays_seconds), 4)
            if replay.release_delays_seconds
            else None
        ),
        merged_ground_truth_events=merged_ground_truth_events,
    )


def compare_thresholds(
    observations: Sequence[FrameObservation],
    thresholds: Iterable[int],
    negative_release_frames: int = 1,
) -> list[ThresholdMetrics]:
    return [
        evaluate_threshold(observations, value, negative_release_frames)
        for value in thresholds
    ]


def compare_temporal_settings(
    observations: Sequence[FrameObservation],
    thresholds: Iterable[int],
    negative_release_values: Iterable[int],
) -> list[ThresholdMetrics]:
    threshold_values = tuple(thresholds)
    return [
        evaluate_threshold(observations, threshold, negative_release_frames)
        for negative_release_frames in negative_release_values
        for threshold in threshold_values
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare temporal thresholds using a saved frame trace"
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--thresholds", default="1,3,5,10")
    parser.add_argument(
        "--negative-release-frames",
        default="1,3",
        help="comma-separated negative-frame release values",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    thresholds = [int(value) for value in args.thresholds.split(",")]
    negative_release_values = [
        int(value) for value in args.negative_release_frames.split(",")
    ]
    results = compare_temporal_settings(
        load_trace(args.trace),
        thresholds,
        negative_release_values,
    )
    rendered = json.dumps([asdict(result) for result in results], indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
