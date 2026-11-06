# DeforTrack V2

> **Current version: V2.** Everything at the root of this repository refers to V2. The first version (V1) is kept in [`v1/`](v1/).

DeforTrack is a multi-source RGB dataset for binary forest / non-forest segmentation. This repository has the split manifests, the trained checkpoints and the code used to evaluate 17 models from the YOLOv8, YOLOv11, Mask R-CNN and U-Net families.

**Dataset:** [DeforTrack on Roboflow (version 3)](https://universe.roboflow.com/eu-sr6tp/defortrack-oqdcu/dataset/3)
**Weights:** [release v2.0](https://github.com/alvarosamp/DeforTrack/releases/tag/v2.0)

## V2 vs. V1

| | V1 | V2 |
| --- | ---: | ---: |
| Train | 12,861 | 12,861 |
| Validation | 895 | 833 |
| Test | 892 | 892 (845 kept + 47 replaced) |
| Cross-split pairs with similarity ≥ 0.98 | See audit stages below | 0 |
| Checkpoint selection | 895 validation images | 833 validation images |

V2 keeps the same training samples and augmentation. Only the validation and test splits changed, and all 17 models were retrained and reevaluated on the V2 split.

## Dataset

| Split | Images |
| --- | ---: |
| Train | 12,861 |
| Validation | 833 |
| Test | 892 |
| **Total** | **14,586** |

Augmentation was applied to training only, with three outputs per source image. The 12,861 training samples come from 4,287 source images. Reference forest masks are the union of the polygons in each image. The 47 replacement test images use masks inherited from Forest Aerial Images.

Before the final split was fixed, images were compared across splits using ResNet-18 embeddings (cosine similarity, threshold 0.98) and SHA-256. The initial audit, before the test replacements, flagged 90 pairs involving test images and 111 validation–training pairs. After the test replacements, a separate audit found no pairs involving the test split at that threshold and flagged 115 validation–training pairs involving 62 distinct validation images. The two validation–training counts come from separate runs on unchanged partitions.

- 47 test images that were too close to training or validation images were replaced with pairs from Forest Aerial Images, keeping the test set at 892 images;
- 62 validation images flagged against training were removed without replacement (895 → 833).

In the final split no pair across splits reaches 0.98 and no file is shared. This screens visual overlap only; it does not guarantee geographic or temporal independence.

[`data/partition_manifest.csv`](data/partition_manifest.csv) lists every image and label in the split with its SHA-256. Use it to check a Roboflow download before reproducing any result.

## Results

All checkpoints were selected on the validation set; the test set was not used for selection. IoU and Dice are per-image means in %. Latency is batch 1 on an RTX 4060 Laptop GPU, including image loading and preprocessing.

| Model | Test IoU | Test Dice/F1 | Forest Aerial IoU | Amazon/Atlantic IoU | Latency (ms) | Weights (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [YOLOv8n](models/yolov8/YOLOv8n/) | 82.54 | 88.77 | 47.56 | 33.86 | 15.0 | 6.8 |
| [YOLOv8s](models/yolov8/YOLOv8s/) | 85.09 | 90.28 | 48.69 | 47.80 | 16.8 | 23.9 |
| [YOLOv8m](models/yolov8/YOLOv8m/) | 85.89 | 90.96 | 47.96 | 40.72 | 25.5 | 54.8 |
| [YOLOv8l](models/yolov8/YOLOv8l/) | 85.64 | 90.59 | 47.15 | 16.64 | 33.4 | 92.3 |
| [YOLOv8x](models/yolov8/YOLOv8x/) | 85.51 | 90.64 | 48.41 | 9.08 | 48.6 | 144.0 |
| [YOLOv11n](models/yolov11/YOLOv11n/) | 84.04 | 89.70 | 51.03 | 20.79 | 13.3 | 6.0 |
| [YOLOv11s](models/yolov11/YOLOv11s/) | 86.04 | 90.95 | 50.04 | 20.76 | 15.3 | 20.6 |
| [YOLOv11m](models/yolov11/YOLOv11m/) | 86.01 | 91.06 | 48.09 | 4.73 | 22.8 | 45.2 |
| [YOLOv11l](models/yolov11/YOLOv11l/) | 86.39 | 91.44 | 48.53 | 7.88 | 25.9 | 55.8 |
| [YOLOv11x](models/yolov11/YOLOv11x/) | 86.50 | 91.29 | 48.58 | 19.01 | 44.6 | 124.8 |
| [mask_rcnn_R_50_C4_1x](models/mask-rcnn/mask_rcnn_R_50_C4_1x/) | 83.09 | 89.42 | 44.36 | 0.26 | 115.8 | 280.3 |
| [mask_rcnn_R_50_C4_3x](models/mask-rcnn/mask_rcnn_R_50_C4_3x/) | 83.41 | 89.53 | 43.02 | 1.48 | 162.9 | 280.3 |
| [mask_rcnn_R_50_DC5_1x](models/mask-rcnn/mask_rcnn_R_50_DC5_1x/) | 84.10 | 89.96 | 44.18 | 4.42 | 155.3 | 1375.1 |
| [mask_rcnn_R_101_C4_3x](models/mask-rcnn/mask_rcnn_R_101_C4_3x/) | 83.39 | 89.52 | 45.69 | 0.23 | 69.8 | 432.3 |
| [mask_rcnn_R_101_DC5_3x](models/mask-rcnn/mask_rcnn_R_101_DC5_3x/) | 84.80 | 90.42 | 46.89 | 5.98 | 66.6 | 1527.1 |
| [mask_rcnn_R_101_FPN_3x](models/mask-rcnn/mask_rcnn_R_101_FPN_3x/) | 84.19 | 90.04 | 46.56 | 10.58 | 69.5 | 503.1 |
| [U-Net](models/u-net/U-Net/) | 76.67 | 85.05 | 45.14 | 2.60 | 82.7 | 37.8 |

Full metrics (precision, recall, Boundary IoU/F1, mask AP, confidence intervals, energy) are in [`results/`](results/), and per-image scores are in each model folder.

## Repository layout

```
data/          split manifest, external dataset manifests, similarity audit
models/        one folder per architecture, one subfolder per checkpoint
  yolov8/  yolov11/  mask-rcnn/  u-net/
annotation/    agreement between the three annotators (248 images)
training/      Colab training pipeline and helpers
results/       consolidated metrics for the 17 checkpoints
benchmark.py   data verification and evaluation
v1/            DeforTrack V1 (first version, 895 validation images)
```

Each checkpoint folder has `training/` (settings and log), `internal/` (test scores), `external/` (Forest Aerial Images and Amazon/Atlantic Forest scores) and `profile/` (latency, memory, energy).

## Usage

Python 3.11. Install the PyTorch build for your CUDA version, then:

```bash
pip install -r requirements.txt
```

Mask R-CNN also needs Detectron2 0.6.

Download the dataset from Roboflow in YOLOv8 format and the weights from the release. Check them with:

```bash
python benchmark.py verify-data test /path/to/defortrack
cd /path/to/downloaded-weights
sha256sum -c /path/to/DeforTrack/models/SHA256SUMS
```

Evaluate a checkpoint:

```bash
python benchmark.py evaluate test /path/to/defortrack \
  --model YOLOv11x --weights /path/to/YOLOv11x.pt --output runs/yolov11x
```

The dataset folder must contain `train/`, `valid/` and `test/`, each with `images/` and `labels/`. For the external sets use `ForestAerial` or `AmazonAtlantic` instead of `test` and point to a folder with `images/` and `masks/`.

Inference uses 640 × 640 images, confidence 0.25 for YOLO and 0.30 for Mask R-CNN, and a 0.50 probability threshold for U-Net. Instance masks are merged into one forest mask before scoring.

## Annotation agreement

Three annotators labeled the same 248 images. Mean pairwise IoU was 75.52% (Dice 84.21%, Cohen's Kappa 0.748). Scores per pair and per image are in [`annotation/`](annotation/).

## Limitations

- The exported images have no EXIF, GPS or acquisition date, and there is no reliable per-image record of region, source or biome.
- The similarity audit does not establish independence by location, source, flight or date.
- Performance drops on external data (best mean IoU 51.03% on Forest Aerial Images and 47.80% on Amazon/Atlantic Forest), so results should not be read as performance on other regions, biomes or sensors.
- Computational measurements describe the evaluation workstation, not embedded hardware.
