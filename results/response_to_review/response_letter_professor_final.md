# Carta-resposta ao professor

Prezado Professor Felipe,

Obrigado pelos comentários. A comparação foi corrigida: todos os modelos da tabela principal de segmentação, incluindo os seis Mask R-CNN, foram avaliados no mesmo subconjunto final de teste com 892 imagens. Os resultados antigos do Mask R-CNN obtidos com 100 imagens foram retirados da tabela principal.

O U-Net também foi reavaliado nas 892 imagens. Os valores confirmados foram IoU de 49,97%, Dice/F1 de 61,80%, precisão de 70,11%, recall de 68,14%, Boundary IoU de 20,93%, Boundary F1 de 34,53% e acurácia de pixel de 72,50%.

O mAP foi mantido apenas para os modelos que produzem instâncias com scores. Como o U-Net produz uma máscara semântica única, seus campos de mAP foram marcados como não aplicáveis, evitando comparar protocolos diferentes.

A tabela computacional foi separada conceitualmente da tabela de acurácia. Nela, `Images` indica o número de amostras usadas no profiling, e não o tamanho do conjunto de teste. O profiling antigo do Mask R-CNN usou 100 imagens; por isso, esses valores não são usados para comparar a acurácia. RAM e CPU foram medidas com `psutil`, e a potência da GPU com NVML.

Também foram corrigidas a descrição da augmentação, a justificativa da união OR, as limitações de generalização, os casos de falha, as referências e a descrição do hardware. Os checkpoints foram treinados em Google Colab com NVIDIA T4, enquanto as reavaliações padronizadas foram executadas localmente em NVIDIA GeForce RTX 4060 Laptop GPU.

Atenciosamente,  
Álvaro Sampaio Careli
