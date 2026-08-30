# Carta-resposta ao professor

**Artigo:** *DeforTrack: A High-Resolution Dataset for Deep Learning-Based Deforestation Segmentation*

Prezado Professor Felipe,

Obrigado pelos comentários e direcionamentos durante a revisão do manuscrito. Abaixo organizo, de forma objetiva, o que foi ajustado no artigo, como cada ponto dos revisores foi tratado e quais aspectos ainda precisam ser assumidos como limitação ou trabalho futuro.

## Resumo das ações realizadas

- A descrição do dataset foi revisada para evitar afirmações não comprovadas sobre superioridade e fontes específicas.
- Foi incluída uma comparação objetiva com outros datasets/plataformas de sensoriamento remoto.
- A discussão dos resultados foi ampliada, incluindo cenários de falha e generalização.
- Foi adicionada avaliação externa com dois datasets semanticamente compatíveis.
- A ausência de inter-annotator agreement formal foi tratada como limitação, com metodologia proposta para versões futuras.
- A questão de leakage espacial/temporal foi discutida com honestidade, destacando que o split foi feito antes da augmentação, mas sem garantia estrita por falta de metadados.

## Pontos tratados

| Ponto | Resposta / ação | Local sugerido |
|---|---|---|
| Comparação com datasets existentes | Foi preparada uma tabela comparativa objetiva com número de imagens/produtos, resolução, tipo de anotação, classes, diversidade geográfica e uso pretendido. | Introdução / Tabela comparativa |
| Afirmação sobre Global Forest Watch | A frase subjetiva sobre falta de nitidez deve ser substituída por uma formulação técnica: produtos GFW baseados em Landsat possuem resolução de 30 m e são menos adequados para pequenos distúrbios e fronteiras locais finas. | Introdução / comparação com datasets |
| Novelty claim | A novidade foi reformulada para não depender de superioridade genérica. O foco passou a ser a combinação de imagens RGB de alta resolução, máscaras manuais, protocolo floresta/não-floresta e avaliação orientada a implantação. | Introdução / contribuições |
| Discussão superficial | A discussão foi ampliada com comportamento por família de modelos, custo computacional, qualidade de borda, cenários de falha e capacidade de generalização. | Results and Discussion / Limitations |
| Casos de falha | Foram descritos cenários prováveis de erro: bordas ambíguas, vegetação esparsa, solo exposto, sombras, áreas agrícolas, vegetação seca, nuvens e baixo contraste. | Limitations and Generalization Considerations |
| Inter-annotator agreement | Como não existe estudo formal nesta versão, isso deve ser declarado como limitação. Foi proposta metodologia futura com anotação independente de amostra representativa, IoU/Dice entre anotadores e adjudicação por especialista. | Limitations / Future work |
| Intervalo de confiança | Foi adicionada a formulação de intervalo de confiança de 95% para métricas por imagem. Se os valores numéricos estiverem disponíveis, eles podem ser incluídos na tabela ou no texto. | Evaluation Metrics |
| Train/val/test split e leakage | Foi esclarecido que o split foi feito antes da augmentação, mas que a falta de metadados completos impede comprovar independência geográfica/temporal estrita. Isso foi tratado como limitação. | Limitations and Generalization Considerations |
| Cross-dataset externo | Foram testados dois datasets externos semanticamente compatíveis e a apresentação foi separada em tabela numérica e figura com exemplos. | External Cross-Dataset Evaluation |

## Pendências que dependem de decisão/orientação

- Definir se será possível realizar uma validação formal das anotações com mais de um anotador.
- Decidir se os intervalos de confiança serão apresentados apenas metodologicamente ou também com valores numéricos.
- Confirmar se há metadados suficientes para algum agrupamento por fonte, região, campanha ou data. Se não houver, manter como limitação.

Atenciosamente,  
Álvaro Sampaio Careli
