# Third-party model

The default runtime uses the **YOLOv8 nano fight/violence checkpoint** published by
[Musawer1214/Fight-Violence-detection-yolov8](https://github.com/Musawer1214/Fight-Violence-detection-yolov8).

- Upstream model: `Yolo_nano_weights.pt`
- Upstream documented classes: Violence/Fight and NoViolence/NoFight
- Upstream documented violence class ID: `1`
- The weight file is **not committed to this repository**. It is downloaded from the upstream source on first use.
- The upstream GitHub README states that the project is MIT-licensed. The matching Hugging Face model card from the same publisher is also tagged MIT.

The pretrained checkpoint was not trained by this repository's author. Model quality depends on the upstream training data and may not generalize to every camera, environment, or type of violent activity.
