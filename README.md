# DeforTrack

Official repository for the DeforTrack binary forest/non-forest segmentation dataset and benchmark. It contains evaluation code, reproducible protocols, trained-model documentation, manuscript sources, and machine-readable results.

## Repository layout

- `article/`: final LaTeX manuscript and bibliography.
- `code/`: internal-test evaluation, profiling, consolidation, external-test, and figure-generation scripts.
- `data/`: dataset access, expected layout, and partition information.
- `models/`: trained-checkpoint names, storage instructions, and expected layout.
- `results/internal_test_892/`: per-model and per-image segmentation results on the common final test subset.
- `results/computational_profile_892/`: standardized local-hardware profiles.
- `results/consolidated/`: publication-ready tables generated from the source result files.
- `results/consolidated/internal_uncertainty_all_models.csv`: means, standard deviations, and 95% confidence intervals for seven metrics and all 17 checkpoints.
- `results/consolidated/external_top3.csv`: ranked Top-3 models for each external benchmark.
- `results/cross_dataset/`: external evaluation outputs.
- `results/split_similarity/`: cross-split ResNet-18 embeddings, cosine-neighbor tables, summary statistics, and visual audit sheet.
- `results/similarity_filtered_test/`: sensitivity analysis after excluding final-test images with train--test cosine similarity of at least 0.98.
- `results/fusion_comparison/`: OR, AND, and confidence-weighted mask-fusion ablation results.
- `results/metadata_audit.json`: complete EXIF, GPS, acquisition-date, and export-sidecar audit.
- `results/environment_local.json`: measured hardware and software environment.
- `results/standardized_protocol_892.json`: machine-readable final protocol.
- `notebooks/`: historical experiment notebooks. Scripts in `code/` and files in `results/` are authoritative for reported values.

## Dataset partitions

| Role in the manuscript | Images |
|---|---:|
| Training | 12,861 |
| Validation | 895 |
| Final test | 892 |

Every internal result in the revised manuscript uses the same 892 held-out final-test images for all 10 YOLO checkpoints, all 6 Mask R-CNN checkpoints, and U-Net. The paper uses these semantic roles independently of directory names inherited from the original export.

## Cross-split visual-similarity audit

The 14,648 images were embedded with the normalized 512-dimensional penultimate representation of an ImageNet-pretrained ResNet-18. Cosine similarity was evaluated exhaustively across training, validation, and test partitions. SHA-256 comparison found no byte-identical cross-split files, but the perceptual audit identified highly similar repeated scenes, overlapping crops, and adjacent frames. In particular, 59 validation images and 45 final-test images had a training neighbor with cosine similarity at or above 0.98. These findings are reported as a limitation rather than treated as proof of geographic or temporal independence.

Reproduce the audit with `code/analyze_split_cosine_similarity.py`. Generate the visual review sheet with `code/make_split_similarity_contact_sheet.py`.

Recompute the similarity-filtered sensitivity analysis with:

```powershell
python .\code\analyze_similarity_filtered_metrics.py
```

At the 0.98 threshold, 45 of the 892 final-test images are excluded and 847 are retained. This diagnostic does not replace a source-, region-, or time-aware test partition.

## Segmentation protocol

Pixel metrics are computed independently per image and macro-averaged over all 892 images: IoU, Dice/F1, precision, recall, pixel accuracy, Boundary IoU, and Boundary F1. Instance-mask mAP is reported for YOLO and Mask R-CNN. U-Net produces a semantic probability mask without instance confidence scores, so instance mAP is not applicable.

For the supplied Mask R-CNN checkpoints, output class identifiers 0 and 1 are legacy synonyms for the same positive forest concept. Accepted masks are merged into one binary forest mask before pixel-level evaluation. This conversion is fixed in the evaluation script and protocol manifest.

The fusion rule was additionally tested with Mask R-CNN ResNet-101 FPN 3x. The weighted-mean threshold was selected on the 895-image validation partition and frozen before the 892-image final-test evaluation. Reproduce the ablation with `code/compare_mask_fusion.py`; machine-readable summary and per-image values are stored in `results/fusion_comparison/`.

## Computational protocol

Every checkpoint is profiled on the same local machine with batch size 1. Before measurement, all 892 files are read once to warm the operating-system cache, and each decoded image is immediately released. This prevents disk-cache order effects without inflating process RAM. Each model then completes one unmeasured full test-set warm-up pass followed by three measured passes, totaling 2,676 synchronized measured inferences.

Latency covers warm-cache image decoding, model-specific preprocessing, GPU inference, CUDA synchronization, and prediction return. CPU utilization is normalized by the 32 logical processors. RAM is process peak resident memory. GPU-board energy is integrated from NVML power samples and reported as joules per image.

## Local environment

The final profiles were run on Windows 11 Education with an Intel Core i9-13900HX (24 cores, 32 logical processors), 16 GB RAM, and an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM. Exact package, driver, CUDA, and library versions are recorded in `results/environment_local.json`.

## Data and trained models

The public dataset is available from the [DeforTrack Roboflow project](https://universe.roboflow.com/alvaro-z5qmu/floresta-wgiln). See `data/README.md` for the expected local layout and the partition roles used by the benchmark.

Trained checkpoints are documented in `models/README.md`. Large model files are not committed through ordinary Git because several checkpoints exceed GitHub's file-size limit. They should be distributed through Git LFS or a versioned GitHub Release.

## Reproduction

Run the standardized profile:

```powershell
.\code\run_all_profiles.ps1
```

Consolidate validated source results:

```powershell
python .\code\consolidate_internal_results.py `
  --yolo-summary .\results\internal_test_892\yolo_test_summary.csv `
  --mask-summary .\results\internal_test_892\maskrcnn_test_summary.csv `
  --mask-map .\results\internal_test_892\maskrcnn_map_892.json `
  --unet-summary .\results\internal_test_892\unet_892_summary.csv `
  --profile .\results\computational_profile_892\standardized_profile_892_v2.csv `
  --out-dir .\results\consolidated
```

Use the environment described in `requirements-evaluation.txt` and `results/environment_local.json`. Paths can be overridden through each script's command-line arguments.

Audit the exported dataset metadata with:

```powershell
python .\code\audit_dataset_metadata.py
```

## Interpretation limits

The audit of all 14,648 exported JPEG images found no EXIF records, GPS coordinates, acquisition timestamps, or authoritative source/biome sidecars. The manuscript therefore does not claim strict source-aware, geographic, temporal, or biome-specific validation. External tests are reported as cross-dataset domain transfer only when their labels are compatible with binary forest/non-forest segmentation.
