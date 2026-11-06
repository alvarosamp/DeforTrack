# YOLOv11

Weights are attached to the [v2.0 release](https://github.com/alvarosamp/DeforTrack/releases/tag/v2.0). Each checkpoint was selected on the 833 validation images.

| Model | Weights | Size (MB) | Test IoU (%) | Test Dice/F1 (%) | SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| [YOLOv11l](YOLOv11l/) | [YOLOv11l.pt](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/YOLOv11l.pt) | 55.8 | 86.39 | 91.44 | `714889b3c789221a997a739f161cdde726741cd8670e598f65ea354adc67804a` |
| [YOLOv11m](YOLOv11m/) | [YOLOv11m.pt](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/YOLOv11m.pt) | 45.2 | 86.01 | 91.06 | `bc9a0d8d6c2389a2a966d0054e5878f785f22a0a5dba68736a7751bb2533cec0` |
| [YOLOv11n](YOLOv11n/) | [YOLOv11n.pt](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/YOLOv11n.pt) | 6.0 | 84.04 | 89.70 | `3c246465f156898887f45dac6560857e86dbcd641ae2c8449ac3aad4f14d37ee` |
| [YOLOv11s](YOLOv11s/) | [YOLOv11s.pt](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/YOLOv11s.pt) | 20.6 | 86.04 | 90.95 | `a54c52c6b1189a4ed1abff388973f14e3f5f78a7c226b7e2057d5cbf6ae3ce6f` |
| [YOLOv11x](YOLOv11x/) | [YOLOv11x.pt](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/YOLOv11x.pt) | 124.8 | 86.50 | 91.29 | `7ef842561fba88fb16023a579a2fa34a5ff6b1b1b510dfdd9a2f3999fa14a836` |

Each model folder contains:

- `training/`: settings and training log of the run
- `internal/`: per-image scores on the 892 test images
- `external/`: per-image scores on Forest Aerial Images and Amazon/Atlantic Forest
- `profile/`: latency, memory and energy on the evaluation workstation
