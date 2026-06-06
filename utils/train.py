"""
utils/train.py — YOLOv8 Fine-tuning Helper

Fine-tunes a YOLOv8 model on your custom violence dataset.

Usage
-----
python utils/train.py \
  --data    path/to/dataset.yaml \
  --model   yolov8m.pt \
  --epochs  50 \
  --imgsz   640 \
  --output  models/violence_yolov8.pt

Dataset YAML format (Roboflow export / custom):
-------------------------------------------------
path: /path/to/dataset
train: images/train
val:   images/val
test:  images/test

nc: 4
names: ['Normal', 'Fighting', 'Weapon', 'Aggression']
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def train(
    data:    str,
    model:   str  = "yolov8m.pt",
    epochs:  int  = 50,
    imgsz:   int  = 640,
    batch:   int  = 16,
    device:  str  = "0",       # "0" = first GPU; "cpu" for CPU-only
    project: str  = "runs/train",
    name:    str  = "violence_detector",
    output:  str  = "models/violence_yolov8.pt",
) -> Path:
    """
    Run YOLOv8 fine-tuning and copy best weights to *output*.
    Returns the path to the saved weights.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

    logger.info("Loading base model: %s", model)
    yolo = YOLO(model)

    logger.info("Starting training — epochs=%d  imgsz=%d  batch=%d", epochs, imgsz, batch)
    results = yolo.train(
        data    = data,
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        device  = device,
        project = project,
        name    = name,
        patience= 15,           # early stopping
        save    = True,
        plots   = True,
    )

    # Copy best weights to project models/ directory
    best_weights = Path(project) / name / "weights" / "best.pt"
    output_path  = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if best_weights.exists():
        shutil.copy(best_weights, output_path)
        logger.info("Best weights saved to: %s", output_path)
    else:
        logger.warning("best.pt not found at %s", best_weights)

    return output_path


def validate(model_path: str, data: str, imgsz: int = 640) -> None:
    """Run validation and print mAP / precision / recall / F1."""
    from ultralytics import YOLO
    yolo   = YOLO(model_path)
    metrics = yolo.val(data=data, imgsz=imgsz)
    logger.info("Validation results:")
    logger.info("  mAP50      : %.4f", metrics.box.map50)
    logger.info("  mAP50-95   : %.4f", metrics.box.map)
    logger.info("  Precision  : %.4f", metrics.box.mp)
    logger.info("  Recall     : %.4f", metrics.box.mr)


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Train YOLOv8 violence detector")
    parser.add_argument("--data",    required=True,          help="Path to dataset.yaml")
    parser.add_argument("--model",   default="yolov8m.pt",   help="Base YOLOv8 weights")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--device",  default="0")
    parser.add_argument("--output",  default="models/violence_yolov8.pt")
    parser.add_argument("--validate", action="store_true",   help="Run validation after training")
    args = parser.parse_args()

    out = train(
        data   = args.data,
        model  = args.model,
        epochs = args.epochs,
        imgsz  = args.imgsz,
        batch  = args.batch,
        device = args.device,
        output = args.output,
    )

    if args.validate:
        validate(str(out), args.data, args.imgsz)
