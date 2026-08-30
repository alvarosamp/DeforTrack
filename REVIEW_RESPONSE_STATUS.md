# Matriz de Resposta da Revisao -- DeforTrack

> STATUS ATUAL: esta matriz foi mantida como histórico. A versão consolidada e
> final das respostas está em `results/response_to_review/response_letter_professor_final.md`
> e `results/response_to_review/response_letter_reviewers_final.md`. A avaliação
> principal agora usa 892 imagens para todos os modelos, incluindo Mask R-CNN.

Este documento distingue alteracoes sustentadas por experimentos executados de
limitacoes que devem permanecer explicitas no manuscrito.

## Professor Felipe -- avaliacao e tabelas

### Particoes, conjunto de teste e comparacao justa

**Resposta proposta:** Concordamos que a versao anterior misturava resultados
calculados sobre diferentes numeros de imagens. A avaliacao principal foi
padronizada para o subconjunto final de 892 imagens. Todos os modelos Mask
R-CNN foram reavaliados nesse mesmo subconjunto usando os pesos treinados,
limiar de confianca 0,30, classes positivas 0 e 1, e mascaras binarias de
floresta obtidas pela uniao das instancias previstas. Assim, IoU, Dice/F1,
precisao, recall, Boundary IoU, Boundary F1 e acuracia de pixel sao diretamente
comparaveis entre YOLO, U-Net e Mask R-CNN. Os resultados de 100 imagens nao
devem permanecer na tabela principal de segmentacao.

**Evidencia executada:** seis Mask R-CNN avaliados em 892 imagens; arquivos em
`outputs/maskrcnn_eval_892` do projeto anterior.

### mAP dos Mask R-CNN

**Resposta proposta:** As colunas mAP@50 e mAP@50--95 serao calculadas por
avaliacao COCO de segmentacao por instancia no mesmo subconjunto de 892
imagens. Os rotulos YOLO sao convertidos para instancias COCO, e as predicoes
das classes positivas sao avaliadas como a classe binaria forest.

**Evidencia executada:** avaliacao COCO finalizada no arquivo
`maskrcnn_map_892.json`. Os resultados foram: R-101 C4 3x (mAP@50 61,33%;
mAP@50--95 39,15%), R-101 DC5 3x (56,22%; 31,80%), R-101 FPN 3x (68,84%;
41,11%), R-50 C4 1x (56,03%; 33,88%), R-50 C4 3x (56,58%; 34,31%) e R-50
DC5 1x (59,02%; 35,09%).

### Metricas computacionais

**Resposta proposta:** A tabela de recursos mede perfil computacional, nao
acuracia. Por isso, deve ser apresentada separadamente da tabela de
segmentacao e a legenda deve declarar que a quantidade de imagens perfiladas
pode diferir. CPU acima de 100% representa a soma de uso em nucleos logicos,
como reportado pelo `psutil`, e nao uma ocupacao superior a capacidade total de
um unico nucleo. A unidade de energia precisa ser corrigida para J/imagem, se
o valor for normalizado pela quantidade de imagens, ou para J/lote, se for o
total de um lote. Nenhuma das duas unidades deve ser inferida sem conferir o
codigo que gerou a tabela.

### Data augmentation

**Resposta proposta:** A configuracao do Roboflow foi definida para gerar tres
saidas aumentadas por exemplo de treinamento. O conjunto exportado final possui
14.648 imagens, incluindo amostras originais e aumentadas. A redacao nao deve
dizer que existem apenas tres imagens no total por exemplo, pois isso seria
ambiguo. As transformacoes sao flip horizontal/vertical, crop de 0--20%, shear
de +/-13 graus, brilho de -22% a +22%, blur ate 3,7 px e ruido em ate 1,56% dos
pixels.

### Fusao de mascaras por OR

**Resposta proposta:** A operacao OR foi escolhida porque cada mascara de
instancia representa uma regiao candidata de floresta. A uniao evita contagem
duplicada de instancias sobrepostas e preserva fragmentos desconectados antes
do calculo da area. Uma operacao AND subestimaria a cobertura ao manter apenas
pixels presentes em todas as mascaras. Uma media ponderada exigiria calibracao
de scores e limiar adicional, nao utilizados no algoritmo de estimativa de area.

### Avaliacao externa

**Resposta proposta:** A avaliacao externa deve informar, para cada conjunto,
numero de imagens, fonte, tipo de imagem, formato de rotulo, conversao para a
tarefa binaria e redimensionamento aplicado. Forest Aerial Images (5.108
imagens) e Amazon/Atlantic Forest (599 imagens) sao os dois conjuntos que devem
permanecer na comparacao principal porque possuem semantica forest/non-forest.
Dead Tree Kaggle e Forest-Change possuem alvos diferentes e devem ser movidos
para material suplementar ou descritos como testes exploratorios, nao como
validacao direta de floresta/não-floresta.

**Evidencia executada:** as avaliacoes externas foram executadas para YOLO,
U-Net e Mask R-CNN em Forest Aerial Images e para YOLO/U-Net nos demais
conjuntos. Os resultados consolidados estao em
`outputs/cross_dataset/additional_cross_dataset_summary.csv`.

## Reviewer 1

### Processo de criacao e superioridade do dataset

**Resposta proposta:** Revisamos a redacao para nao alegar superioridade geral
do processo de criacao. A contribuicao passa a ser apresentada como um conjunto
heterogeneo para segmentacao binaria de floresta/não-floresta, acompanhado de
tabela comparativa objetiva com numero de imagens, resolucao, classes,
anotacao em nivel de pixel, cobertura geografica documentada e uso pretendido.
Quando metadados de origem nao existirem no export final, a coluna deve indicar
"not available in the final export", e nao fazer uma alegacao geografica sem
evidencia. A frase sobre falta de nitidez do Global Forest Watch deve ser
removida, a menos que seja sustentada por uma fonte especifica e verificavel.

### Novidade e discussao de falhas

**Resposta proposta:** A alegacao de novidade foi reduzida a elementos
verificaveis: imagens RGB de multiplas fontes, rotulacao pixel a pixel,
avaliacao de arquiteturas e perfil computacional. Foram adicionados exemplos de
sucesso e falha. A discussao relaciona falsos positivos e falsos negativos a
sombra, variacao de iluminacao, baixo contraste, vegetacao esparsa ou seca,
solo exposto, agricultura, estradas e fronteiras ambiguas.

### Qualidade das anotacoes

**Resposta proposta:** Incluimos estudo de acordo entre tres anotadores em 248
imagens. O IoU pareado medio foi 74,55%, Dice/F1 medio 83,58% e Kappa medio
0,769. Os intervalos de confianca do IoU devem ser descritos como IC de 95% da
media, e os desvios-padrao de Dice/F1 e precisao somente devem ser incluidos se
calculados a partir dos valores por imagem.

### Variancia e repetibilidade

**Resposta proposta:** Para cada metrica por imagem, reportaremos media,
desvio-padrao e IC de 95% na tabela suplementar. A tabela principal pode manter
os valores medios para legibilidade e referenciar a tabela suplementar. Isso
nao substitui repeticoes de treinamento com sementes diferentes; tal experimento
so deve ser alegado se realmente executado.

## Reviewer 3 -- split e leakage

**Resposta proposta:** Nao afirmamos ausencia comprovada de leakage espacial ou
temporal. O split foi executado antes da augmentacao, o que reduz replicacao
direta de imagens aumentadas entre subconjuntos. Contudo, o export final nao
preserva metadados completos de fonte, regiao, data ou campanha de aquisicao.
Assim, nao foi possivel realizar um split geograficamente, temporalmente ou por
fonte estrito, nem comprovar independencia espacial/temporal. Os resultados
sao interpretados como benchmark interno no DeforTrack. Versoes futuras devem
preservar metadados por imagem e aplicar particionamento source-aware,
region-wise e temporal.

## Itens que nao devem ser prometidos sem novo dado

- Cobertura precisa por bioma, estado, regiao ou origem de cada imagem, pois o
  export final nao preserva todos os metadados por imagem.
- Ausencia comprovada de leakage espacial ou temporal.
- Repetibilidade entre diferentes sementes de treinamento, caso os treinos nao
  sejam repetidos.
- Superioridade geral sobre datasets existentes.
