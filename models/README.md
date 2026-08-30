# Trained checkpoints

The benchmark evaluates the following trained checkpoints:

- YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, and YOLOv8x
- YOLOv11n, YOLOv11s, YOLOv11m, YOLOv11l, and YOLOv11x
- Mask R-CNN ResNet-101 C4 3x, DC5 3x, and FPN 3x
- Mask R-CNN ResNet-50 C4 1x, C4 3x, and DC5 1x
- U-Net

Some Mask R-CNN checkpoints exceed 1 GB and cannot be stored through ordinary
GitHub Git objects. Publish the checkpoint bundle through Git LFS or a versioned
GitHub Release, preserving the filenames expected by the evaluation scripts.

Do not commit local virtual environments, framework caches, or duplicate model
exports. The exact model sizes used by the article are recorded in
`results/consolidated/computational_metrics_892.csv`.
