# Third-party model

The default runtime uses the **YOLOv8-nano fight/violence checkpoint** published by [Musawer1214/Fight-Violence-detection-yolov8](https://github.com/Musawer1214/Fight-Violence-detection-yolov8).

- Upstream model file: `Yolo_nano_weights.pt`
- Pinned upstream commit: `20f0d05054cff7da2dc78dee3c2de1bd54106a13`
- Observed/documented classes: `non_violence` and `violence`
- Violence class ID used by this project: `1`
- The weight file is **not committed to this repository**; `utils/download_model.py` downloads the pinned upstream file into `models/`.
- The upstream README describes the project as MIT-licensed, although consumers should review the upstream repository/model terms themselves before redistribution or commercial use.

The pretrained checkpoint was not trained by this repository's author. This repository does not claim model accuracy or ownership of the upstream training data. Model quality can vary with camera angle, lighting, scene type, and other conditions.
