# Response Letter to the Reviewers

Dear Reviewers,

We sincerely thank the reviewers for their constructive comments. The manuscript was revised to clarify the dataset contribution, correct the evaluation protocol, expand the discussion, and state the limits of the available metadata.

## Reviewer 1

**Dataset creation and unsupported superiority claim.** We removed the broad superiority claim and retained an objective comparison based on image quantity, resolution, annotation type, classes, available geographic information, and intended use. DeforTrack is presented as a task-specific forest/non-forest benchmark with 6,094 original RGB images, 640 x 640 patches, polygon-based pixel annotations, and 14,648 exported images after augmentation.

**Global Forest Watch statement.** The subjective statement about lack of sharpness at high zoom levels was removed. The revised text compares the documented spatial scale and intended use of the cited products with the manually annotated RGB patches in DeforTrack.

**Novelty and discussion.** The contribution is now restricted to verifiable characteristics: multi-source RGB imagery, pixel-level masks, quantitative forest-cover statistics, and a benchmark covering segmentation and computational metrics. The discussion now covers shadows, illumination, low contrast, sparse or dry vegetation, exposed soil, agricultural regions, fragmented boundaries, and domain shift.

**Annotation quality.** We added an inter-annotator study on 248 images labeled independently by three annotators. Mean pairwise IoU was 74.55 +/- 19.44%, Dice/F1 was 83.58 +/- 16.64%, and mean pixel-level Kappa was 0.727 +/- 0.216. The limitations of image-level Kappa under class imbalance are reported.

**Variance and repeatability.** Image-level variability and the 95% confidence-interval procedure are described. We do not claim repeatability across training seeds because repeated training runs were not performed.

## Reviewer 3

**Split and leakage.** The manuscript now states that the principal results for YOLOv8, YOLOv11, Mask R-CNN, and U-Net use the same final test subset of 892 images. The dataset roles are reported as 12,861 training, 895 validation, and 892 final test images. The split preceded augmentation, reducing direct leakage from augmented copies. Complete spatial, temporal, source-aware, and biome-wise independence cannot be proven because the final export lacks complete image-level metadata; this is explicitly treated as a limitation.

## Standardized model comparison

All six supplied Mask R-CNN checkpoints were rerun on the same 892-image test subset. U-Net was also rerun on all 892 images. The previous Mask R-CNN values based on 100 images were removed from the main segmentation table. U-Net mAP is marked as not applicable because U-Net outputs a semantic mask rather than scored object instances.

Sincerely,  
Álvaro Sampaio Careli
