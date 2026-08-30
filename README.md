# DeforTrack

Reproducibility repository for the DeforTrack forest/non-forest segmentation
article.

## Repository layout

- `article/`: manuscript source and article-specific correction notes.
- `code/`: evaluation and figure-generation scripts.
- `notebooks/`: original training and metric-extraction notebooks.
- `results/internal_test_892/`: standardized Mask R-CNN and U-Net evaluations
  on the final 892-image test subset.
- `results/cross_dataset/`: external evaluation summaries.
- `results/response_to_review/`: reviewer and professor response material.
- `figures/`: qualitative figures used by the manuscript, when available.

## Dataset and models

The raw dataset and trained weights are kept outside this Git repository because
the local archives and model files are large. The paths used for the local
experiments were:

- Dataset: `D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3`
- Models and auxiliary data: `D:\Datasets\Defortrack`

The final internal test evaluation used 892 images. The source export contains
12,861 training images, 895 validation images, and 892 final test images. The
paper refers to these roles consistently, independently of the directory names
used by the export.

## Verified internal evaluations

All six supplied Mask R-CNN checkpoints were evaluated on the same 892-image
test subset. The evaluation used Detectron2, a confidence threshold of 0.30,
positive classes 0 and 1 merged into a binary forest mask, and COCO-style mask
mAP. Pixel metrics were computed from the union of predicted forest masks.

The supplied U-Net checkpoint was also evaluated on all 892 images. Its verified
means are: IoU 49.97%, Dice/F1 61.80%, precision 70.11%, recall 68.14%,
Boundary IoU 20.93%, Boundary F1 34.53%, and pixel accuracy 72.50%.

U-Net produces a semantic probability mask rather than scored object
instances. Therefore, U-Net mAP is not reported in the revised instance-mAP
columns unless a separate, explicitly defined semantic AP protocol is added.

## Reproduction

Run from the repository root with the project environment that contains
Detectron2, PyTorch, Keras, OpenCV, and the required metric packages. Example:

```powershell
python evaluate_unet_defortrack.py `
  --images "D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3\valid\images" `
  --labels "D:\Datasets\Alvaro\Floresta-3-20260714T215823Z-1-001\Floresta-3\valid\labels" `
  --model "D:\Datasets\Defortrack\unet_instance_segmentation_best.keras" `
  --out-dir results/internal_test_892/unet
```

The supplied dataset directory is used as the final 892-image test subset for
the manuscript, as documented above. Do not add raw archives, virtual
environments, caches, or checkpoints to Git unless a separate storage policy is
defined.

## Current limitations recorded for the paper

The exported dataset does not preserve complete image-level source, geographic,
biome, campaign, and acquisition-date metadata. Consequently, the paper does
not claim strict source-aware, geographic, temporal, or biome-wise validation.
External tests are described as cross-dataset evaluations only when their target
semantics are compatible with binary forest/non-forest segmentation.
