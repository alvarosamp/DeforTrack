# U-Net

Weights are attached to the [v2.0 release](https://github.com/alvarosamp/DeforTrack/releases/tag/v2.0). Each checkpoint was selected on the 833 validation images.

| Model | Weights | Size (MB) | Test IoU (%) | Test Dice/F1 (%) | SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| [U-Net](U-Net/) | [U-Net.keras](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/U-Net.keras) | 37.8 | 76.67 | 85.05 | `7c762b68b05b893d625d2673378e98799ab81ad7e22c87b4d3a6481b8c984bb9` |

Each model folder contains:

- `training/`: settings and training log of the run
- `internal/`: per-image scores on the 892 test images
- `external/`: per-image scores on Forest Aerial Images and Amazon/Atlantic Forest
- `profile/`: latency, memory and energy on the evaluation workstation
