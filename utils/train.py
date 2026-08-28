"""Optional YOLOv8 fine-tuning helper. Training is not required for the default setup."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def train(data: str, model: str = "yolov8m.pt", epochs: int = 50, imgsz: int = 640, batch: int = 16, device: str = "0", project: str = "runs/train", name: str = "violence_detector", output: str = "models/violence_yolov8.pt") -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics not installed. Run: pip install -r requirements.txt") from exc
    yolo = YOLO(model)
    yolo.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device, project=project, name=name, patience=15, save=True, plots=True)
    save_dir = Path(yolo.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise RuntimeError(f"Training completed but best.pt was not found at {best_weights}")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output_path)
    logger.info("Best weights copied to %s", output_path)
    return output_path


def validate(model_path: str, data: str, imgsz: int = 640) -> None:
    from ultralytics import YOLO
    metrics = YOLO(model_path).val(data=data, imgsz=imgsz)
    logger.info("mAP50: %.4f", metrics.box.map50)
    logger.info("mAP50-95: %.4f", metrics.box.map)
    logger.info("Precision: %.4f", metrics.box.mp)
    logger.info("Recall: %.4f", metrics.box.mr)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Optional YOLOv8 fine-tuning helper")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default="models/violence_yolov8.pt")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    out = train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.device, output=args.output)
    if args.validate:
        validate(str(out), args.data, args.imgsz)
