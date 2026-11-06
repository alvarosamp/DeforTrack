# External cross-dataset results

These files record inference-only evaluation of DeforTrack-trained checkpoints
on semantically compatible external forest/non-forest datasets. No external
sample was used for DeforTrack training or model selection.

- `forest_aerial/`: 5,108 paired 256 x 256 RGB images and binary masks.
- `amazon_atlantic/`: 599 paired 512 x 512 Sentinel-2 samples. Only the RGB
  bands were retained; the NIR band was discarded for the RGB-trained models.
- `examples/`: representative images used by the manuscript.
- `cross_dataset_all_models_summary.csv`: consolidated Forest Aerial results.
- `additional_cross_dataset_summary.csv`: consolidated additional-dataset
  results, including Amazon/Atlantic Forest.

The folders contain summary and per-image CSV files only. Dataset copies, image
caches, checkpoints, and intermediate exports are intentionally excluded.
