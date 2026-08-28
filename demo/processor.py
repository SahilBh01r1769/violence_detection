"""Video analysis helpers for the hosted Streamlit demo."""

from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from core.detector import ViolenceDetector

_DETECTOR_LOCK = threading.Lock()


@dataclass
class VideoAnalysisSummary:
    frames_analyzed: int
    violent_samples: int
    alerts_triggered: int
    peak_confidence: float
    duration_seconds: float
    analysis_fps: float
    timeline: list[dict]
    alert_frames: list[np.ndarray]
    output_video: Path | None


def _safe_fps(value: float) -> float:
    return value if value and 0 < value < 240 else 30.0


def analyse_video(
    video_path: str | Path,
    detector: ViolenceDetector,
    *,
    target_analysis_fps: float = 4.0,
    max_duration_seconds: float = 30.0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> VideoAnalysisSummary:
    """Run the existing detector on sampled frames from a video file."""

    source = Path(video_path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the selected video.")

    input_fps = _safe_fps(float(capture.get(cv2.CAP_PROP_FPS)))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    stride = max(1, int(round(input_fps / max(0.5, target_analysis_fps))))
    max_source_frames = max(1, int(max_duration_seconds * input_fps))
    source_frames_to_consider = (
        min(total_frames, max_source_frames) if total_frames > 0 else max_source_frames
    )
    estimated_samples = max(1, (source_frames_to_consider + stride - 1) // stride)

    temp_output = tempfile.NamedTemporaryFile(
        prefix="violence_demo_output_", suffix=".mp4", delete=False
    )
    output_path = Path(temp_output.name)
    temp_output.close()

    writer = None
    if frame_width > 0 and frame_height > 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            max(1.0, input_fps / stride),
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            writer.release()
            writer = None
            output_path.unlink(missing_ok=True)

    timeline: list[dict] = []
    alert_frames: list[np.ndarray] = []
    frames_analyzed = 0
    violent_samples = 0
    alerts_triggered = 0
    peak_confidence = 0.0
    source_frame_index = 0

    with _DETECTOR_LOCK:
        # The detector is cached by Streamlit to avoid reloading PyTorch weights
        # on every rerun. Reset only its per-video mutable temporal state here.
        detector._window.clear()
        detector._frame_id = 0

        try:
            while source_frame_index < max_source_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                if source_frame_index % stride != 0:
                    source_frame_index += 1
                    continue

                result = detector.process_frame(frame)
                annotated = detector.annotate_frame(result)
                if writer is not None:
                    writer.write(annotated)

                frames_analyzed += 1
                violent_samples += int(result.is_violent)
                alerts_triggered += int(result.alert_triggered)
                peak_confidence = max(peak_confidence, result.max_confidence)
                time_s = source_frame_index / input_fps

                timeline.append(
                    {
                        "time_s": round(time_s, 2),
                        "violent": int(result.is_violent),
                        "confidence": round(result.max_confidence, 4),
                        "class_name": result.primary_class,
                        "alert": bool(result.alert_triggered),
                    }
                )

                if result.alert_triggered and len(alert_frames) < 6:
                    alert_frames.append(annotated.copy())

                if progress_callback:
                    progress_callback(frames_analyzed, estimated_samples)

                source_frame_index += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()

    if frames_analyzed == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("No video frames could be analyzed.")

    if writer is None or not output_path.exists() or output_path.stat().st_size == 0:
        output_path = None

    duration_seconds = min(source_frame_index / input_fps, max_duration_seconds)
    actual_analysis_fps = (
        frames_analyzed / duration_seconds if duration_seconds > 0 else 0.0
    )

    return VideoAnalysisSummary(
        frames_analyzed=frames_analyzed,
        violent_samples=violent_samples,
        alerts_triggered=alerts_triggered,
        peak_confidence=peak_confidence,
        duration_seconds=duration_seconds,
        analysis_fps=actual_analysis_fps,
        timeline=timeline,
        alert_frames=alert_frames,
        output_video=output_path,
    )
