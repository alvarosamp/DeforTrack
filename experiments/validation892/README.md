# Expanded validation (892 images)

This experiment re-evaluates the 17 checkpoints from [release v2.0](https://github.com/alvarosamp/DeforTrack/releases/tag/v2.0). The checkpoints were selected on the original 833-image validation split and were not reselected on the expanded split. No retrained weights enter these results.

| Split | Released V2 | Expanded experiment |
| --- | ---: | ---: |
| Train | 12,861 | 12,861 |
| Validation | 833 | 892 |
| Internal test | 892 | 892 |
| Total | 14,586 | 14,645 |

The 59 additions are from Forest Aerial Images and appear only in validation. Their reference masks were converted to YOLO polygons (mean conversion IoU 99.68%, minimum 99.09%). The conversion check does not validate the forest labels. Nine additions are [flagged for semantic review](data/review_record.json). The expanded experiment is not the downloadable Roboflow v3 release.

The [partition manifest](data/partition_manifest.csv) records the SHA-256 of each image and label. The [similarity audit](data/partition_protocol.json) found zero cross-split pairs at or above 0.98 and no byte-identical cross-split images. These checks do not establish geographic or temporal independence. Mean reference forest cover is 42.37% (population SD 26.64 percentage points).

## Results

- [Validation metrics for all 17 checkpoints](results/validation_summary.csv), [833 vs. 892 comparison](results/validation_comparison.csv) and [per-image scores](results/per_image/). Mean validation IoU fell by 3.04 to 3.47 percentage points across the checkpoints when the 59 pairs were included. YOLOv11x remained first: 89.86% on the original 833, 41.51% on the 59 additions alone, and 86.66% on the expanded split. These are changes in the evaluation population, not changes in model weights or internal test performance.
- [External rankings](results/external_rankings.csv). Forest Aerial Images retains 4,963 of the 5,022 previously evaluated pairs after excluding the validation additions. YOLOv11n leads with 51.15% mean IoU. Amazon/Atlantic Forest retains all 599 pairs, and YOLOv8s leads with 47.80%. These results were aggregated from the published per-image scores using the retained image and mask hashes.
- [Internal test region metrics](results/internal_test_region_unchanged.csv). The images, masks, and checkpoints are unchanged. Published internal instance AP and computational profiles remain in the [main results directory](../../results/).
- [Fusion calibration](results/fusion_selection.json) and [test results](results/fusion_summary.csv). Recalibrating the weighted rule on the 892 validation images still selects 0.05. The original checkpoint remains fixed.
- [LaTeX excerpts](results/article_verified_tables.tex) for annotation agreement, split similarity, forest cover, and the external ranking.

## Reproduce

From the repository root, recompute the external ranking and internal region summaries without images or weights:

```bash
python experiments/validation892/reevaluate.py --stage aggregate --output runs/validation892-aggregate
python experiments/validation892/reevaluate.py --stage comparison --output runs/validation892-comparison
```

To repeat validation inference and the fusion comparison, use CUDA, the [repository dependencies](../../README.md#usage), the released checkpoints, and a local copy of the expanded dataset. The script checks every validation and test pair against the manifest and verifies each checkpoint hash. The data folder must contain `valid/images`, `valid/labels`, `test/images`, and `test/labels`.

```bash
python experiments/validation892/reevaluate.py --stage validation \
  --data /path/to/expanded-defortrack --weights /path/to/release-weights \
  --output runs/validation892-inference
```

The manifests and [source-pair inventory](data/selected_sources.csv) are included here. Images, masks and checkpoint files are not redistributed in this branch.
