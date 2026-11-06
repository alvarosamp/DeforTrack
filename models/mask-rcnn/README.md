# Mask R-CNN (Detectron2)

Weights are attached to the [v2.0 release](https://github.com/alvarosamp/DeforTrack/releases/tag/v2.0). Each checkpoint was selected on the 833 validation images.

| Model | Weights | Size (MB) | Test IoU (%) | Test Dice/F1 (%) | SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| [mask_rcnn_R_101_C4_3x](mask_rcnn_R_101_C4_3x/) | [mask_rcnn_R_101_C4_3x.pth](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/mask_rcnn_R_101_C4_3x.pth) | 432.3 | 83.39 | 89.52 | `59de9c5c5cc9548af626740c5116e9954daf3c7a40bb0e4be324abb0da3b76f5` |
| [mask_rcnn_R_101_DC5_3x](mask_rcnn_R_101_DC5_3x/) | [mask_rcnn_R_101_DC5_3x.pth](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/mask_rcnn_R_101_DC5_3x.pth) | 1527.1 | 84.80 | 90.42 | `82fc17daa074249164ad6a13a223188b3bf6cd00f8d794335f3b91465f826c19` |
| [mask_rcnn_R_101_FPN_3x](mask_rcnn_R_101_FPN_3x/) | [mask_rcnn_R_101_FPN_3x.pth](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/mask_rcnn_R_101_FPN_3x.pth) | 503.1 | 84.19 | 90.04 | `b08bebd872802e8c4ca3d8919e821801c1ecec4fd1cdcaa781113932b767078d` |
| [mask_rcnn_R_50_C4_1x](mask_rcnn_R_50_C4_1x/) | [mask_rcnn_R_50_C4_1x.pth](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/mask_rcnn_R_50_C4_1x.pth) | 280.3 | 83.09 | 89.42 | `05bc4cd16a307434c2bbf8f161f50aa0ba36f607953bb61a8c97ad66f66213b3` |
| [mask_rcnn_R_50_C4_3x](mask_rcnn_R_50_C4_3x/) | [mask_rcnn_R_50_C4_3x.pth](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/mask_rcnn_R_50_C4_3x.pth) | 280.3 | 83.41 | 89.53 | `f9967d7ec9de57d1af0d053216f5594d67f0d82921de3be477eb3c6b4eae8b2a` |
| [mask_rcnn_R_50_DC5_1x](mask_rcnn_R_50_DC5_1x/) | [mask_rcnn_R_50_DC5_1x.pth](https://github.com/alvarosamp/DeforTrack/releases/download/v2.0/mask_rcnn_R_50_DC5_1x.pth) | 1375.1 | 84.10 | 89.96 | `dae5044806eb3b82155fcc81f0afa162115de61e4eb32ef4c2117d14f9953c9b` |

Each model folder contains:

- `training/`: settings and training log of the run
- `internal/`: per-image scores on the 892 test images
- `external/`: per-image scores on Forest Aerial Images and Amazon/Atlantic Forest
- `profile/`: latency, memory and energy on the evaluation workstation
