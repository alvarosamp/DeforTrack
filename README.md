# DeforTrack

Official repository for the DeforTrack binary forest/non-forest segmentation dataset and benchmark. It contains reproducible evaluation code, trained-model documentation, manuscript sources, and machine-readable results.

## Repository layout

- `article/`: LaTeX manuscript and bibliography.
- `code/`: similarity audit, test-set construction, evaluation, profiling, and consolidation scripts.
- `data/`: dataset access and expected local layout.
- `models/`: checkpoint names and storage instructions.
- `results/replacement_test_892/`: visual audit, replacement manifest, final similarity audit, per-image metrics, and consolidated tables for the corrected final test.
- `results/cross_dataset/`: external evaluation outputs.
- `results/split_similarity/`: original exhaustive cross-split embedding audit.
- `results/metadata_audit.json`: EXIF, GPS, acquisition-date, and sidecar audit.

Large datasets and checkpoints are not stored in ordinary Git history. See `data/README.md` and `models/README.md`.

## Dataset partitions

| Role | Images |
|---|---:|
| Training | 12,861 |
| Validation | 895 |
| Similarity-safe final test | 892 |

The final test contains 845 retained images and 47 replacements. Every reported internal segmentation result uses this same 892-image set for all 10 YOLO checkpoints, all 6 Mask R-CNN checkpoints, and U-Net.

## Similarity-safe final test

Images were represented by normalized 512-dimensional penultimate-layer embeddings from an ImageNet-pretrained ResNet-18. The initial exhaustive audit identified 90 train-test or validation-test pairs with cosine similarity at or above 0.98, involving 47 unique test images. Manual inspection confirmed that all 47 represented the same scene, an overlapping crop, a rotation, or an adjacent frame.

Those images were replaced from a held-out candidate pool while preserving the test size. Candidates were required to:

- have no source identifier already represented in the released splits;
- have cosine similarity below 0.98 to every existing train, validation, and original-test image;
- have pairwise cosine similarity below 0.98 among selected replacements;
- closely match the removed image's forest-cover ratio;
- preserve the raster mask with mask-to-polygon round-trip IoU of at least 0.98.

The final audit found zero SHA-256 duplicates and zero cosine-similarity pairs at or above 0.98 between the recomposed test and either training or validation. The observed maxima were 0.979895 against training and 0.978390 against validation.

Reproduce the workflow with:

```powershell
python .\code\make_threshold_similarity_audit.py --help
python .\code\select_similarity_safe_replacements.py --help
python .\code\build_replacement_test_set.py --help
python .\code\audit_replacement_test_similarity.py --help
```

Portable visual-review tables, manifests, contact sheets, and final audit files are under `results/replacement_test_892/`.

## Evaluation protocol

No model was retrained for the replacement analysis. Each existing checkpoint was evaluated with its fixed inference threshold. Pixel metrics are computed per image and macro-averaged: IoU, Dice/F1, precision, recall, pixel accuracy, Boundary IoU, and Boundary F1. Instance-mask mAP@50 and mAP@50--95 are reported for YOLO and Mask R-CNN. U-Net is a semantic model and therefore has no instance-level mAP.

The Mask R-CNN fusion ablation selects the weighted-mean threshold only on the 895-image validation set and applies the frozen threshold to the similarity-safe test. OR, AND, and weighted-mean outputs are available under `results/replacement_test_892/fusion_comparison/`.

## Computational protocol

Every checkpoint was reprofiled on the similarity-safe final test with batch size 1 on the same local workstation. The profiler performs one full unmeasured 892-image warm-up pass and three complete measured passes, totaling 2,676 synchronized inferences per checkpoint. Latency includes image decoding, preprocessing, GPU inference, CUDA synchronization, and prediction return. GPU energy is integrated from NVML board-power samples and reported in joules per image.

The profiling workstation used Windows 11 Education, an Intel Core i9-13900HX, 16 GB RAM, and an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM. These measurements characterize this workstation and are not embedded-device benchmarks.

## Main outputs

- `results/replacement_test_892/consolidated/segmentation_metrics_892.csv`
- `results/replacement_test_892/consolidated/computational_metrics_892.csv`
- `results/replacement_test_892/model_metrics/`
- `results/replacement_test_892/fusion_comparison/`
- `results/replacement_test_892/final_audit/`

Run the table consistency check with:

```powershell
python .\code\audit_table_consistency.py `
  --tex .\article\DeforTrack_revised.tex `
  --segmentation .\results\replacement_test_892\consolidated\segmentation_metrics_892.csv `
  --computational .\results\replacement_test_892\consolidated\computational_metrics_892.csv
```

Run the standardized computational profile after copying
`models/profile_models.example.json` to a local manifest and replacing its
checkpoint paths:

```powershell
.\code\run_all_profiles.ps1 `
  -Python C:\path\to\python.exe `
  -Images C:\path\to\replacement_test_892\images `
  -ModelsJson C:\path\to\profile_models.local.json
```

## Data and checkpoints

The public dataset is available from the [DeforTrack Roboflow project](https://universe.roboflow.com/alvaro-z5qmu/floresta-wgiln). Trained checkpoints should be distributed through Git LFS or a versioned release because several files exceed GitHub's ordinary file-size limit.

## Interpretation limits

The similarity audit addresses visual overlap at the prespecified 0.98 threshold, but does not establish geographic or temporal independence. The exported files do not preserve authoritative image-level coordinates, dates, biome labels, or complete upstream provenance. Results therefore characterize the stated partitioning protocol and should not be interpreted as evidence of performance in unseen regions, dates, biomes, or sensors.
