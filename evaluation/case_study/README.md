# Two-clip temporal-filter case study

This case study tests how detector confidence, positive-frame qualification,
and negative-frame release affect event output. It is a small controlled
example, not an accuracy benchmark.

## Source clips and annotations

| Trace | Source clip | Decoded frames | Source FPS | Human annotation |
| --- | --- | ---: | ---: | --- |
| `violence_trace.csv` | [Strong female mixed martial arts fighter](https://mixkit.co/free-stock-video/strong-female-mixed-martial-arts-fighter-40991/) | 694 | 24 | Entire decoded clip is violent |
| `nonviolence_trace.csv` | [People having a work meeting around a table](https://mixkit.co/free-stock-video/people-having-a-work-meeting-around-a-table-4547/) | 848 | 30 | No violent interval |

Both source pages identify the downloads as free stock video under the Mixkit
Stock Video Free License. The annotations were made by reviewing the clips,
not by copying the detector output. The traces contain frame decisions rather
than the source videos.

## Detector and timestamp provenance

- Trace exporter revision: `85b3618728abef679d9440109857224a73f83953`.
- Model source: `Musawer1214/Fight-Violence-detection-yolov8` at upstream
  revision `20f0d05054cff7da2dc78dee3c2de1bd54106a13`.
- Checkpoint filename: `violence_yolov8n.pt`, downloaded from the pinned
  `Yolo_nano_weights.pt` URL configured in `config.py`.
- Violence mapping: class ID `1`; configured names include `violence`,
  `fight`, `fighting`, and `violence/fight`.
- Capture confidence floor: 0.25. A zero confidence means no violence-class
  box survived that floor, so these traces cannot replay thresholds below
  0.25.
- Timestamps are video time, calculated as `(frame_id - 1) / source_fps`.
  They do not measure inference or notification wall-clock latency.

The nonviolent trace covers video timestamps 0.0000-28.2333 seconds. The
violent trace covers 0.0000-28.8750 seconds. The final decoded frame is part
of each annotation.

## Controlled comparison

`C` is detector confidence, `N` is consecutive positive frames required to
start an event, and `K` is consecutive negative frames required to release
the active-event latch. The two result JSON files contain every combination
of C = 0.40, 0.55, 0.70, 0.85; N = 1, 3, 5, 10; and K = 1, 3.

Selected rows show the main trade-off:

| C | N | K | False events on meeting clip | Triggers in violent interval | Duplicate triggers | First-alert delay (video seconds) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.40 | 1 | 1 | 29 | 81 | 80 | 0.0000 |
| 0.40 | 5 | 3 | 11 | 13 | 12 | 0.1667 |
| 0.55 | 5 | 1 | 5 | 10 | 9 | 1.7917 |
| 0.55 | 5 | 3 | 5 | 8 | 7 | 1.7917 |
| 0.70 | 5 | 1 | 5 | 8 | 7 | 1.7917 |
| 0.70 | 5 | 3 | 3 | 8 | 7 | 1.7917 |
| 0.70 | 10 | 3 | 1 | 1 | 0 | 28.0833 |
| 0.85 | 1 | 1 | 0 | 0 | 0 | missed |

No tested setting both eliminated false events and detected the violent clip.
N=10 reduced fragmentation but delayed the first alert until 28.0833 seconds
at C=0.70. C=0.85 removed false events by producing no event on either clip.

At C=0.55 and N=5, changing K from 1 to 3 reduced violent-clip triggers from
10 to 8 but left the meeting clip at 5 false events. At C=0.70 and N=5, K=3
reduced meeting-clip false events from 5 to 3 compared with K=1. K=3 released
two frame intervals after the first negative: 0.0667 seconds at 30 FPS and
0.0833 seconds at 24 FPS.

## Provisional operating point

The application defaults to C=0.70, N=5, and K=3. This is provisional. On
these traces it improves false-event and fragmentation counts over the former
C=0.55, N=5, K=1 defaults without increasing the measured first-alert delay.
It still produced three false events in one short nonviolent clip and seven
duplicate triggers within one continuous violent interval.

This sample cannot measure whether K=3 merges distinct real events because it
contains only one ground-truth violent interval. Broader claims would require
more independently annotated clips from different settings.

## Reproduce trace capture and JSON results

Download the clips to `sample_videos/violence.mp4` and
`sample_videos/nonviolence.mp4`, then run from the repository root:

```bash
python -m evaluation.capture_trace sample_videos/nonviolence.mp4 evaluation/case_study/nonviolence_trace.csv --capture-confidence-floor 0.25
python -m evaluation.capture_trace sample_videos/violence.mp4 evaluation/case_study/violence_trace.csv --ground-truth 0:29 --capture-confidence-floor 0.25
```

Replay the saved detections without loading YOLO again:

```bash
python -m evaluation.temporal evaluation/case_study/nonviolence_trace.csv --confidence-thresholds 0.40,0.55,0.70,0.85 --thresholds 1,3,5,10 --negative-release-frames 1,3 --output evaluation/case_study/nonviolence_results.json
python -m evaluation.temporal evaluation/case_study/violence_trace.csv --confidence-thresholds 0.40,0.55,0.70,0.85 --thresholds 1,3,5,10 --negative-release-frames 1,3 --output evaluation/case_study/violence_results.json
```

The test suite independently replays both committed traces and requires the
result objects to match the committed JSON exactly. Line endings may differ
between operating systems.
