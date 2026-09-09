import csv
import json

from evaluation.capture_trace import file_sha256, write_capture_metadata
from evaluation.report import build_rows, write_csv, write_svg


def test_report_and_capture_metadata(tmp_path):
    trace = tmp_path / "clip.csv"
    trace.write_text(
        "frame_id,timestamp_seconds,is_violent,confidence,ground_truth_violent,capture_confidence_floor\n"
        "1,0.0,True,0.8,True,0.25\n2,0.1,True,0.8,True,0.25\n",
        encoding="utf-8",
    )
    rows = build_rows([("clip", trace)], [0.5, 0.7], [1, 2], [1, 3])
    output_csv, output_svg = tmp_path / "summary.csv", tmp_path / "tradeoff.svg"
    write_csv(rows, output_csv)
    write_svg(rows, output_svg)
    with output_csv.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 8
    assert "Temporal-filter trade-off" in output_svg.read_text(encoding="utf-8")

    video, model = tmp_path / "video.mp4", tmp_path / "model.pt"
    video.write_bytes(b"video")
    model.write_bytes(b"model")
    metadata = json.loads(write_capture_metadata(tmp_path / "trace.csv", video, model).read_text())
    assert metadata["source_video_sha256"] == file_sha256(video)
    assert metadata["model_sha256"] == file_sha256(model)
    assert set(metadata["packages"]) == {"ultralytics", "opencv-python", "numpy"}
