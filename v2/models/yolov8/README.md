# YOLOv8

Weights are attached to the [v2.0 release](https://github.com/alvarosamp/DeforTrack/releases/tag/v2.0). Each checkpoint was selected on the 833 validation images.

| Model | Weights | Size (MB) | Test IoU (%) | Test Dice/F1 (%) | SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| [YOLOv8l](YOLOv8l/) | [YOLOv8l.pt](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/YOLOv8l.pt) | 92.3 | 85.64 | 90.59 | `4ff1d99bff87783d6a88e9d97d0d203ebf15d05c09ce5fb6cb18e5fdaf05981d` |
| [YOLOv8m](YOLOv8m/) | [YOLOv8m.pt](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/YOLOv8m.pt) | 54.8 | 85.89 | 90.96 | `6d98fbf7d8c46a623a948704c9aa6245e1a0c91671ab737a4a301458d18f2797` |
| [YOLOv8n](YOLOv8n/) | [YOLOv8n.pt](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/YOLOv8n.pt) | 6.8 | 82.54 | 88.77 | `3d065bdcda312cd35ed44d4d0a4b29810e2cb0ce4de5ffc2983c9eafae06bae2` |
| [YOLOv8s](YOLOv8s/) | [YOLOv8s.pt](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/YOLOv8s.pt) | 23.9 | 85.09 | 90.28 | `d771b089e4d571132d12c6a3441c609b57e8f4c7a24a41d43f3c8798d05d63b9` |
| [YOLOv8x](YOLOv8x/) | [YOLOv8x.pt](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/YOLOv8x.pt) | 144.0 | 85.51 | 90.64 | `644f84c59304cc0f4e5b75c600137993273f181d8415b022ddf3a864f290e77c` |

Each model folder contains:

- `training/`: settings and training log of the run
- `internal/`: per-image scores on the 892 test images
- `external/`: per-image scores on Forest Aerial Images and Amazon/Atlantic Forest
- `profile/`: latency, memory and energy on the evaluation workstation
