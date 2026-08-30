# Response Letter to the Reviewers

**Manuscript:** *DeforTrack: A High-Resolution Dataset for Deep Learning-Based Deforestation Segmentation*

Dear Reviewers,

We sincerely thank the reviewers for their careful reading of the manuscript and for the constructive comments. The manuscript was revised to improve the justification of the dataset contribution, clarify the evaluation protocol, expand the discussion of limitations and failure cases, and provide a more transparent treatment of generalization and data leakage risks.

Below, we summarize the main changes made in response to each comment.

## Point-by-point response

| Reviewer | Comment | Response / manuscript change |
|---|---|---|
| Reviewer 1 | Dataset creation process is common practice; superiority over existing datasets not justified. | We added an objective comparison table including number of images/products, spatial resolution, annotation type, classes/labels, geographic diversity, and intended use. We revised the text to avoid an unsupported broad superiority claim and to define DeforTrack as a task-specific benchmark for high-resolution forest/non-forest segmentation. |
| Reviewer 1 | The statement about Global Forest Watch requires a reference. | We revised the wording to a technically supported statement: GFW tree-cover products commonly rely on Landsat-based 30 m data, which are less suitable for small disturbances and fine-grained local boundaries. The revised statement is supported by GFW/WRI documentation on tree-cover datasets and spatial resolution. |
| Reviewer 1 | Novelty claim not rigorously supported. | We strengthened the novelty discussion by linking the contribution to measurable dataset characteristics: 6,094 original images, 14,648 images after augmentation, 640 x 640 RGB patches, manual polygon-based masks, binary forest/non-forest labels, multiple acquisition sources, and deployment-oriented benchmarking. |
| Reviewer 1 | Discussion remains superficial. | We expanded the discussion to include model-family behavior, accuracy-efficiency trade-offs, boundary quality, computational constraints, external generalization, and likely failure scenarios such as ambiguous forest boundaries, sparse vegetation, exposed soil, shadows, clouds, agricultural regions, dry vegetation, and low-contrast scenes. |
| Reviewer 1 | No inter-annotator agreement or labeling quality assessment. | We acknowledge that a formal inter-annotator agreement study was not performed in the current version. The limitation is now explicitly stated, and future work will include independent annotation of a representative subset, mask-level IoU/Dice agreement analysis, and expert adjudication of disagreement cases. |
| Reviewer 1 | Table I overwhelming; no variance/repeatability. | We reorganized the results into clearer segmentation and computational metric tables and added the 95% confidence interval formulation for image-level metrics to improve repeatability reporting. |
| Reviewer 3 | Train/validation/test split and leakage. | We clarified that the split was performed before data augmentation, reducing direct leakage from augmented copies. However, because the final exported dataset does not preserve complete image-level source, region, date, or flight-campaign metadata, strict geographic/temporal leakage analysis could not be fully guaranteed. This is now discussed as a limitation and future work. |

## Additional clarification on Global Forest Watch

We revised the wording related to Global Forest Watch to avoid a subjective statement. Instead of stating that it "lacks sharpness at higher zoom levels," the manuscript now refers to the spatial-resolution limitation more precisely: Global Forest Watch tree-cover products commonly rely on Landsat-based 30 m resolution data, which are less suitable for identifying small disturbances or fine-grained local forest boundaries.

This statement is supported by GFW/WRI documentation describing 30 m tree-cover products and explaining that 10 m products improve monitoring at small local scales.

## Remaining limitations explicitly acknowledged

- A formal inter-annotator agreement study was not performed in the current version.
- Strict geographic or temporal split validation could not be fully guaranteed because complete image-level source metadata were not preserved in the final exported dataset.
- The external datasets provide complementary evidence of generalization under domain shift, but they do not replace a fully source-aware cross-dataset validation protocol.

Sincerely,  
Álvaro Sampaio Careli
