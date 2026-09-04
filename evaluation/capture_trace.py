"""Run video inference once and save a replayable frame-level trace."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

import cv2

from config import CONFIDENCE_THRESHOLD, MODEL_PATH
from core.detector import DetectionResult, ViolenceDetector


@dataclass(frozen=True)
class TimeInterval:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class TraceSummary:
    frames: int
    positive_frames: int
    source_fps: float
    duration_seconds: float
    output_path: str


class FrameDetector(Protocol):
    def process_frame(self, frame: object) -> DetectionResult: ...


def parse_interval(value: str) -> TimeInterval:
    try:
        start_text, end_text = value.split(":", maxsplit=1)
        start, end = float(start_text), float(end_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "ground-truth intervals must use START:END seconds"
        ) from exc

    if not math.isfinite(start) or not math.isfinite(end):
        raise argparse.ArgumentTypeError("interval boundaries must be finite")
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError(
            "an interval must satisfy 0 <= START < END"
        )
    return TimeInterval(start, end)


def is_ground_truth_violent(
    timestamp_seconds: float, intervals: Iterable[TimeInterval]
) -> bool:
    return any(
        interval.start_seconds <= timestamp_seconds < interval.end_seconds
        for interval in intervals
    )


def export_trace(
    video_path: Path,
    output_path: Path,
    ground_truth: Sequence[TimeInterval],
    *,
    confidence: float = CONFIDENCE_THRESHOLD,
    detector: FrameDetector | None = None,
    capture_factory: Callable[[str], object] = cv2.VideoCapture,
) -> TraceSummary:
    """Export detector decisions using half-open ground-truth intervals.

    ``confidence`` in the output is the strongest violent-class detection on the
    frame. It is zero when the detector finds no violent-class box.
    """
    capture = capture_factory(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"Video reports an invalid frame rate: {fps}")

    frame_detector = detector or ViolenceDetector(
        MODEL_PATH,
        confidence=confidence,
        frame_consistency=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".part")
    frame_count = 0
    positive_frames = 0

    try:
        with partial_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "frame_id",
                    "timestamp_seconds",
                    "is_violent",
                    "confidence",
                    "ground_truth_violent",
                ],
            )
            writer.writeheader()

            while True:
                available, frame = capture.read()
                if not available:
                    break
                frame_count += 1
                timestamp = (frame_count - 1) / fps
                result = frame_detector.process_frame(frame)
                violent_confidence = max(
                    (
                        detection.confidence
                        for detection in result.violent_detections
                    ),
                    default=0.0,
                )
                positive_frames += int(result.is_violent)
                writer.writerow(
                    {
                        "frame_id": frame_count,
                        "timestamp_seconds": f"{timestamp:.6f}",
                        "is_violent": result.is_violent,
                        "confidence": f"{violent_confidence:.6f}",
                        "ground_truth_violent": is_ground_truth_violent(
                            timestamp, ground_truth
                        ),
                    }
                )
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    finally:
        capture.release()

    return TraceSummary(
        frames=frame_count,
        positive_frames=positive_frames,
        source_fps=round(fps, 4),
        duration_seconds=round(frame_count / fps, 4),
        output_path=str(output_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a video once and export decisions for temporal replay"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--ground-truth",
        action="append",
        default=[],
        type=parse_interval,
        metavar="START:END",
        help="violent interval in seconds; repeat for separate events",
    )
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    summary = export_trace(
        args.video,
        args.output,
        args.ground_truth,
        confidence=args.confidence,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
