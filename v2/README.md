# DeforTrack V2

[Roboflow v3](https://universe.roboflow.com/eu-sr6tp/defortrack-oqdcu/dataset/3) lists 12,861 training, 833 validation, and 892 test images. The split used in the study is defined by [partition_manifest.csv](data/partition_manifest.csv). It records the SHA-256 of each image and its YOLO annotation. Raw images are not included here. Check the downloaded files against the manifest before reproducing a run.

| Architecture | Checkpoints | Files |
| --- | ---: | --- |
| YOLOv8 | 5 | [Models and results](models/yolov8/) |
| YOLOv11 | 5 | [Models and results](models/yolov11/) |
| Mask R-CNN | 6 | [Models and results](models/mask-rcnn/) |
| U-Net | 1 | [Model and results](models/u-net/) |

The [checkpoint index](models/checkpoints.csv) records asset names, file sizes, and digests. The weights are distributed separately from Git history. Each was selected on the 833 validation images; the test set was not used for selection. The weights are attached to the [v2.0 release](https://github.com/alvarosamp/DeforTrack/releases/tag/v2.0) and can be checked with `sha256sum -c models/SHA256SUMS`.

## Verify data and run evaluation

Use Python 3.11 with the [evaluation requirements](../requirements-evaluation.txt). Install the PyTorch build matching your CUDA environment. Mask R-CNN also requires Detectron2 0.6.

~~~bash
python v2/benchmark.py verify-data test /path/to/defortrack
python v2/benchmark.py evaluate test /path/to/defortrack \
  --model YOLOv8s --weights /path/to/YOLOv8s.pt --output run/yolov8s
~~~

For internal data, the root contains `train/`, `valid/`, and `test/`, each with `images/` and `labels/`. For external evaluation, use `ForestAerial` or `AmazonAtlantic` instead of `test`. Point the command at the external dataset directory containing `images/` and `masks/`. The external manifests list the 5,022 screened Forest Aerial pairs and the 599 Amazon/Atlantic pairs.

The evaluator checks file hashes, uses 640-pixel inference, and reports image-level metrics and macro averages. It uses confidence thresholds of 0.25 for YOLO and 0.30 for Mask R-CNN, and a pixel probability threshold above 0.50 for U-Net. Instance-mask AP on the internal test is computed for YOLO and Mask R-CNN only.

## Annotation agreement

[annotation/](annotation/) holds the matching manifest, the 744 pairwise scores, the scores against the two-of-three consensus and the 248 per-image averages. `interannotator.py` recomputes them from the three annotation exports.

## Training and results

Model directories contain the available training settings and logs, internal and external per-image scores, and computational profiles. Machine-specific paths were removed from the settings files. The [training code](training/) contains the shared Colab pipeline and its helpers. The saved settings and logs document the evaluated runs, including model-specific differences from the initial queue.

- [Internal metrics](results/verified_internal_892.csv)
- [External metrics](results/verified_external.csv)
- [Computational profiles](results/verified_profiles.csv)

The computational results were measured on the RTX 4060 Laptop GPU workstation used in the study. The similarity audit screens visual overlap but does not establish geographic or temporal independence.
