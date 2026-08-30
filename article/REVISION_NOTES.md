# Article revision notes

The manuscript copy in `DeforTrack_revised.tex` incorporates the corrections
supported by the local files and completed evaluations.

- The principal segmentation table reports the same final test subset of 892
  images for YOLO, Mask R-CNN, and U-Net.
- All six Mask R-CNN checkpoints were rerun on those 892 images. Their COCO
  mask mAP values are stored in `results/internal_test_892/maskrcnn_map_892.json`.
- U-Net was rerun on all 892 images. Its per-image metrics, means, standard
  deviations, and 95% confidence intervals are stored in
  `results/internal_test_892/unet_892_per_image.csv` and
  `results/internal_test_892/unet_892_summary.csv`.
- U-Net mAP is marked as not applicable in the instance-mAP columns because
  U-Net outputs a semantic probability mask, not scored object instances.
- The paper no longer claims strict geographic, temporal, biome-wise, or
  source-aware generalization without the required image-level metadata.
- The resource table remains a profiling table. Its `Images` column means the
  number of images used during profiling and must not be interpreted as the
  sample size of the accuracy benchmark.

The source export contains 12,861 training images, 895 validation images, and
892 final test images. The paper uses those roles consistently even if the
local export directory names do not reflect the intended roles.
