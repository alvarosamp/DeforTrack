# Resposta aos revisores e resumo das modificações

Artigo: *DeforTrack: A High-Resolution Dataset for Deep Learning-Based Deforestation Segmentation*

## Carta de agradecimento

Prezados Revisores,

Gostaríamos de agradecer sinceramente pelo tempo dedicado à leitura do manuscrito e pelos comentários detalhados. As observações recebidas foram fundamentais para aprimorar a clareza metodológica, a organização dos resultados, a discussão sobre generalização e a apresentação visual do artigo. Revisamos o manuscrito considerando cada ponto levantado e descrevemos abaixo as principais modificações realizadas.

As alterações foram feitas com o objetivo de tornar o texto mais preciso, evitar afirmações não comprovadas, melhorar a rastreabilidade dos experimentos e apresentar de forma mais honesta as limitações da validação atual.

## Resumo geral das modificações

- A descrição do conjunto de dados foi revisada para remover afirmações específicas sobre fontes públicas que não deveriam ser explicitamente citadas no texto principal.
- A seção de Data Augmentation foi renomeada para *Data Augmentation and Availability* e recebeu o rótulo de referência cruzada correspondente.
- Foi adicionada uma explicação sobre o procedimento usado para medir RAM, CPU, tempo de inferência, FPS e consumo de energia.
- As tabelas de métricas foram revisadas para destacar em negrito os melhores valores de desempenho e eficiência computacional.
- O parágrafo de resultados referente aos modelos YOLOv11 foi corrigido para ficar consistente com os valores oficiais das tabelas.
- Foi adicionada uma avaliação externa com dois datasets adicionais, separando os resultados numéricos das imagens de exemplo para melhorar a legibilidade.
- A limitação relacionada à ausência de validação cross-dataset estrita foi explicitada de forma curta e honesta.
- As referências bibliográficas foram revisadas para incluir os datasets externos usados na nova avaliação.

## Resposta ponto a ponto

| Comentário / ponto revisado | Modificação realizada | Local no manuscrito |
|---|---|---|
| Clareza sobre as fontes do dataset | O texto foi ajustado para descrever as três fontes principais de forma mais geral: UAV, Google Earth Engine e repositórios públicos ambientais. Foram removidas menções específicas que não procediam no contexto final do artigo. | Dataset Implementation - Data Collection |
| Disponibilidade do dataset e referência cruzada | A subseção passou a se chamar *Data Augmentation and Availability* e foi adicionado o label `sec:data_augmentation` para corrigir a referência interna do texto. | Dataset Implementation - Data Augmentation and Availability |
| Procedimento de medição computacional | Foi acrescentado que RAM, CPU, tempo de inferência, FPS e energia foram medidos pelo mesmo procedimento para todos os modelos. Também foi informado o uso de `psutil` e NVML. | Deep-learning Semantic Segmentation Models Evaluation - Evaluation Metrics |
| Consistência dos resultados YOLOv11 | O parágrafo de discussão dos YOLOv11 foi reescrito para refletir os valores oficiais das tabelas, destacando YOLOv11m para IoU, YOLOv11x para métricas de contorno e YOLOv11n para compacidade. | Results and Discussion |
| Destaque dos melhores valores nas tabelas | As melhores métricas de segmentação e eficiência computacional foram destacadas em negrito, incluindo IoU, mAP, Dice/F1, Boundary IoU/F1, Pixel Accuracy, Precision, Recall, tamanho, RAM, tempo, FPS, CPU e energia. | Tables: Segmentation metrics and Computational metrics |
| Cross-dataset validation | Foi esclarecido que a validação interna original não constitui validação cross-dataset estrita. Para responder ao comentário, foram realizados testes externos em dois datasets adicionais semanticamente compatíveis. | Results and Discussion - External Cross-Dataset Evaluation |
| Apresentação dos datasets externos | A tabela externa foi simplificada para conter apenas os resultados numéricos e interpretações. As imagens representativas foram movidas para uma figura separada, aumentando a legibilidade. | Table: External cross-dataset evaluation; Figure: external dataset examples |
| Limitações e generalização | Foi adicionada uma limitação curta explicando que a validação cross-dataset estrita permanece como trabalho futuro devido à ausência de metadados de origem por imagem no dataset exportado. | Limitations and Generalization Considerations |
| Referências | As referências foram revisadas para incluir os datasets externos usados na nova avaliação e para manter coerência entre as citações do texto e a bibliografia. | References |

## Texto sugerido para acompanhar a submissão

We thank the reviewers for their careful reading of the manuscript and for their constructive comments. The manuscript was revised to improve the dataset description, clarify the evaluation protocol, correct inconsistencies in the reported results, and provide a more transparent discussion of generalization limitations. In particular, we revised the data collection description, added details about computational profiling, corrected the YOLOv11 discussion according to the final metrics, highlighted the best values in the tables, and added an external evaluation using two additional forest segmentation datasets. We also explicitly state that strict cross-dataset validation remains future work due to the absence of image-level source metadata in the final exported dataset.

## Observação final

A versão revisada evita afirmar que os resultados internos demonstram generalização cross-dataset estrita. Em vez disso, o manuscrito apresenta os resultados internos como benchmark do DeforTrack e os novos testes externos como evidência complementar de generalização parcial sob mudança de domínio.

## Complemento: resposta aos pontos adicionais de revisão

### Reviewer 1

**Comment: Dataset creation process is common practice; superiority over existing datasets not justified.**

**Response:** We thank the reviewer for this important observation. In the revised manuscript, we avoided relying on a broad or unsupported claim of superiority and instead added an objective comparison between DeforTrack and representative remote sensing datasets/platforms. The new comparison table includes the number of images or products, spatial resolution, annotation type, classes/labels, geographic diversity, and intended use. This revision clarifies that DeforTrack's contribution is not the use of standard dataset construction procedures themselves, but the combination of high-resolution RGB image patches, manual polygon-based pixel-level masks, a binary forest/non-forest protocol, and deployment-oriented benchmarking.

**Change in the manuscript:** A comparative dataset table was added to the Introduction, and the discussion was revised to explain the specific gap addressed by DeforTrack.

**Additional note:** The statement about Global Forest Watch should not be written as "lacks sharpness at higher zoom levels" unless directly supported by a source. A more precise and referencable formulation is: "Global Forest Watch tree-cover products commonly rely on Landsat-based 30 m resolution data, which are less suitable for identifying small disturbances or fine-grained local forest boundaries." This can be supported by Global Forest Watch/WRI material describing the 30 m resolution and its reduced granularity for small disturbances.

---

**Comment: Novelty claim not rigorously supported.**

**Response:** We agree that the novelty claim required stronger support. The revised manuscript now supports the contribution through objective comparison with existing resources and by emphasizing measurable dataset characteristics, including the number of original and augmented images, image resolution, annotation type, binary segmentation protocol, and source diversity. The novelty claim was reformulated to avoid overstatement and to focus on DeforTrack as a task-specific benchmark for high-resolution forest/non-forest semantic segmentation.

**Change in the manuscript:** The Introduction and contribution list were revised to define the dataset contribution more precisely. The dataset comparison table and dataset characterization strengthen the justification of the novelty claim.

---

**Comment: Discussion remains superficial.**

**Response:** We thank the reviewer for this comment. The Results and Discussion section was expanded to provide a more detailed interpretation of model behavior. The revised text discusses performance trade-offs between YOLO, Mask R-CNN, and U-Net, including segmentation accuracy, boundary quality, computational cost, and suitability for embedded or offline use. We also added a limitations/generalization discussion covering ambiguous boundaries, fragmented landscapes, sparse vegetation, exposed soil, agricultural regions, shadows, dry vegetation, cloud interference, and low-contrast areas as likely failure cases.

**Change in the manuscript:** The Results and Discussion and Limitations and Generalization Considerations sections were expanded to discuss failure scenarios, model limitations, and generalization capacity.

---

**Comment: No inter-annotator agreement or labeling quality assessment.**

**Response:** We agree that annotation quality assessment is important for strengthening the dataset description. In the current version, the annotations were manually produced and reviewed, but a formal inter-annotator agreement study was not performed. To address this limitation transparently, the revised manuscript now states this as a limitation and proposes an annotation quality assessment protocol for future versions. This protocol will include independent review of a representative subset of images, calculation of agreement metrics such as IoU/Dice between annotators, and adjudication of disagreement cases by a senior reviewer.

**Change in the manuscript:** A limitation was added noting that a formal inter-annotator agreement analysis was not performed in the current release. Future work now includes independent annotation review and quantitative labeling quality assessment.

**Suggested manuscript text:** "Although all masks were manually annotated and reviewed, a formal inter-annotator agreement study was not performed in the current version. Future releases will include independent annotation of a representative subset of images and agreement analysis using mask-level IoU/Dice scores, followed by expert adjudication of disagreement cases."

---

**Comment: Table I overwhelming; no variance/repeatability.**

**Response:** We thank the reviewer for pointing this out. To improve repeatability analysis, the revised manuscript includes a description of 95% confidence intervals for the main image-level metrics. The confidence interval is computed from the distribution of metric values across validation images using the standard formulation based on the sample mean, standard deviation, and number of evaluated images.

**Change in the manuscript:** A confidence interval formulation was added to the Evaluation Metrics section. The tables were also reorganized into segmentation metrics and computational metrics to improve readability.

### Reviewer 3

**Comment: Train/val/test split and leakage.**

**Response:** We agree that data leakage and split independence are critical for evaluating remote sensing datasets. The revised manuscript clarifies that the current split was performed before data augmentation, reducing direct leakage from augmented copies of the same images. However, we also acknowledge that the current exported dataset does not preserve image-level source metadata sufficient to guarantee strict geographic, temporal, or source-aware separation. Therefore, the reported results should be interpreted as an internal benchmark on DeforTrack rather than as proof of strict cross-region or cross-time generalization.

**Change in the manuscript:** A limitation was added explaining that strict geographic/temporal split validation was not possible in the current exported version due to the absence of complete image-level source metadata. Future work will preserve acquisition source, region, biome, flight campaign, and acquisition date metadata to enable source-aware, geographic, and temporal validation protocols.

**Suggested manuscript text:** "Although the split was performed before data augmentation, the current exported dataset does not preserve complete image-level metadata regarding source, geographic region, acquisition date, or flight campaign. Therefore, strict spatial/temporal leakage analysis and region-wise validation could not be fully guaranteed. Future versions of DeforTrack will preserve this metadata to enable geographic, temporal, and source-aware validation protocols."
