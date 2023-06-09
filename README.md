<h1 align="center">Detecção de Fraude de Cartão de Crédito 💳</h1>

## Problema de Negócio

**Contexto:** Um ponto de extrema importância para as empresas de cartão de crédito é a capacidade de reconhecer transações fraudulentas, a fim de evitar que os clientes sejam cobrados por itens que não compraram. Sabendo disso, este projeto empregará algoritmos de Machine Learning para detectar transações de crédito fraudulentas.

## Sobre o conjunto de dados
O conjunto de dados contém transações feitas por cartões de crédito em setembro de 2013 por titulares de cartões europeus. Este conjunto de dados apresenta transações ocorridas em dois dias, onde existem **492 fraudes em 284.807 transações**. O conjunto de dados é altamente desbalanceado, a classe positiva (fraudes) responde por 0,172% de todas as transações.

As variáveis de entrada numéricas neste projeto são resultados de uma transformação **PCA (Principal Component Analysis)**. Por questões de confidencialidade, não são fornecidos os recursos originais dos dados. As características V1, V2, …, V28 representam os principais componentes obtidos por meio do PCA. As únicas características que não foram submetidas à transformação PCA são "Time" e "Amount".

O recurso "Time" indica o tempo decorrido em segundos entre cada transação e a primeira transação no conjunto de dados. Já o recurso "Amount" representa o valor da transação. Por fim, a característica "Classe" é a variável de resposta, assumindo o **valor 1 em caso de fraude e 0 caso contrário**.

O dataset está disponível publicamente no Kaggle. **Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Skills
- Python;
- Pandas;
- Numpy;
- Matplotlib;
- Seaborn;
- Scikit-learn;
- TensorFlow;
- Keras;
- Machine Learning;

## Processo de análise
Neste projeto, foram utilizados os modelos KNN e Decision Tree para detectar fraudes em transações de cartão de crédito, por meio da análise de uma base de dados de fraude de cartão de crédito. O projeto foi conduzido em diversas etapas, que incluem:

![apresentacao](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/3ec20637-d24a-4e10-81e6-0968e55abbc7)

As etapas do projeto foram as seguintes:

- **Plano de Negócio:** Compreender o contexto e os objetivos para solucionar o problema.
- **Análise exploratória dos dados:** O objetivo dessa etapa foi entender o comportamento dos dados, verificar a distribuição das classes, identificar possíveis falhas no conjunto de dados, como duplicações ou valores ausentes. Além disso, exploramos visualmente as relações entre as variáveis, incluindo a correlação, e examinamos as distribuições de tempo e valor.
- **Pré-processamento dos dados:** Não foi necessário realizar uma limpeza profunda dos dados, pois não foram encontrados valores ausentes ou duplicados. No entanto, foi necessário lidar com o desbalanceamento dos dados, pois a maioria das transações eram não fraudulentas. Para isso, aplicamos as técnicas de RandomUnderSampling (RUS) e SMOTE para balancear o conjunto de dados, garantindo a qualidade dos dados de treinamento para os modelos de Machine Learning.
- **Treinamento dos modelos:** Foi realizado o treinamento dos modelos Decision Tree e KNN, que serviram como máquinas preditivas para resolver o problema em questão. Utilizamos a validação cruzada K-fold para avaliar o desempenho dos modelos e estimar sua capacidade de generalização para dados não vistos.
- **Análise dos resultados:** Por fim, para analisar os resultados obtidos, foram utilizadas as métricas acurácia, precisão, recall e F1-Score. Além disso, foram plotadas a matriz de confusão, a Curva ROC e calculamos a área sob a curva (AUC), as quais são medidas importantes em problemas de classificação binária.

## Análise Exploratória dos Dados
Nesta etapa, realizamos uma análise exploratória para examinar e estudar as características do conjunto de dados. Inicialmente, verificamos as informações estatísticas dos valores e se o conjunto de dados apresenta valores ausentes ou duplicados.

Posteriormente, traçamos gráficos para compreender o comportamento das variáveis. Ao analisar a distribuição dos dados em cada classe, constatamos que o conjunto de dados está desbalanceado. Observamos que existem 284.315 amostras para transações não fraudulentas, enquanto apenas 492 amostras correspondem a transações fraudulentas. Essa discrepância indica uma distorção no conjunto de dados.

![download](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/e92e01ea-bd52-4450-88be-d13d3c5869d6)

Além disso, foi realizada uma análise das associações entre as variáveis numéricas por meio de uma matriz de correlação. Essa análise permite observar o grau de correlação entre as variáveis. É possível notar que os atributos "V2" e "Amount" apresentam uma correlação negativa, indicando uma relação inversa entre eles. Por outro lado, os atributos "V7" e "Amount" possuem uma correlação positiva, sugerindo uma relação direta entre eles. Outro fato importante é que as variáveis não apresentam alta correlação entre si, visto que, a correlação forte entre as variáveis pode trazer desafios no treinamento dos modelos, podendo fornecer informações redundantes e resultar em *overfitting*, influenciando diretamente no desempenho do modelo.

![corr](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/aeafb528-919d-4558-9853-6c5ef1d2ac24)

Por fim, realizou-se uma análise das distribuições das variáveis "Time" e "Amount" em relação aos grupos de transações normais e fraudulentas. Observa-se que as transações fraudulentas têm uma concentração maior de valores entre 0€ e 1.000€, enquanto as transações normais estão distribuídas entre 0€ e 5.000€. Quanto ao atributo "Time", não se observa diferença perceptível entre os dois tipos de transações.

![amout](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/d1495dd2-2d06-436c-bb82-d398f85fbcbb)

## Pré-processamento dos Dados
Nesta etapa, foi realizado o balanceamento dos dados para garantir a qualidade dos dados de treinamento do modelo. Utilizaram-se duas técnicas: **RandomUnderSampling (RUS) e SMOTE**. A técnica RUS reduz a quantidade de exemplos da classe majoritária, selecionando aleatoriamente uma amostra desses exemplos. Isso resulta em um conjunto de dados balanceado, porém, com uma quantidade reduzida de dados. Já a técnica SMOTE gera exemplos sintéticos para a classe minoritária. O SMOTE é especialmente útil quando a classe minoritária apresenta regiões com poucos exemplos, permitindo preencher essas regiões com exemplos sintéticos.

Dessa forma, ao utilizar a técnica RUS, houve uma redução na quantidade de dados de treinamento, sendo utilizados 369 exemplos para a classe de transações não fraudulentas e 369 exemplos para as transações fraudulentas.

![grafico_rus](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/31ab23a5-5858-4604-a57e-458f3dc61298)

Em contrapartida, com técnica SMOTE, foram gerados dados sintéticos de treinamento, totalizando 213.236 amostras para a classe de transações não fraudulentas e 213.236 para as transações fraudulentas.

![grafico_Smote](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/1029b6b1-541b-4c47-ad76-23ee952e1d62)

## Treinamento dos modelos

Durante o treinamento dos modelos Decision Tree e KNN, foi utilizado a técnica GridSearchCV, que auxilia na determinação de valores ideais para os hiperparâmetros e no controle da complexidade do modelo. Essaa abordagem nos permitiu ajustar os parâmetros de forma sistemática, explorando diferentes combinações e avaliando o desempenho do modelo em cada configuração. Foram definidos os seguites hiperparâmetros para cada modelo:

Decision Tree  | max_depth | min_samples_split | min_samples_leaf 
--------- | -------- | -------- | -------- 
RUS | 5 | 1 | 2
SMOTE | None |	1	| 2

KNN  | n_neighbors | weights 
--------- | -------- | -------- 
RUS | 5 |	distance |
SMOTE | 3 |	distance |	

Além disso, foi utilizada a validação cruzada K-fold (K=5) para avaliar o desempenho dos modelos e estimar a capacidade de generalização em dados não vistos.

## Análise dos Resultados
Para analisar os resultados obtidos, foram utilizadas métricas como acurácia, precisão, recall e F1-Score. Além disso, fez-se uso da matriz de confusão, da Curva ROC e do valor da área sob a curva (AUC), medidas importantes em problemas de classificação binária. 

Após avaliar os resultados, foi constatado que a técnica que o modelo Decision Tree, treinado com o conjunto de dados balanceados utilizando a técnica SMOTE, apresentou os resultados mais satisfatórios para o problema em questão. A Tabela 1 apresenta os resultados para as métricas Acurácia, Precisão, Recall e F1-Score - obtidas com cada técnica utilizada no estudo. Os resultados fornecem informações valiosas sobre o desempenho dos modelos e auxiliam na seleção da abordagem mais eficaz para a detecção de fraudes em transações.

Modelo   | Acurácia | Precisão | Recall | F1-Score | AUC
--------- | -------- | -------- | -------- | -------- | --------
Decision Tree - RUS | 92.39 | [99.98, 2.0] | [92.4, 89.43] | [96.04, 3.9] | 90.92 |
**Decision Tree - SMOTE** | **99.78** |	[99.97, 42.56]	| [99.8, 83.74] |	[99.89, 56.44]	 | **91.77** |
KNN - RUS | 65.03 |	[99.91, 0.32] |	[65.03, 65.85] |	[78.78, 0.65] |	65.44
KNN - SMOTE | 95.10 |	[99.91, 1.75] |	[95.18, 49.59] |	[97.48, 3.38]	| 72.38

Ademais, a imagem a seguir apresenta a matriz de confusão para todos os modelos treinados. É possível observar que o modelo Decision Tree (para os dados balanceados com a técnica SMOTE), consegue acertar 70.940 amostras de transações não fraudulentas (Verdadeiro Positivo) e 103 amostras de transações fraudulentas (Verdadeiro Negativo), errando 139 (Falso Positivo) e 20 (Falso Negativo) amostras respectivamente. 

![matriz](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/455ab9ba-4e5c-4505-abde-820fadcef720)

Por fim, a curva ROC para cada modelo é plotada no gráfico a seguir. Novamente, o modelo Decision Tree apresenta a melhor configuração para os valores da taxa de falsos positivos (FPR) e da taxa de verdadeiros positivos (TPR), apresentados no gráfico. Em conformidade, o modelo apresenta um valor para área sob a curva ROC (AUC) de 97.77%.
 
![curva_roc](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/056da439-c00a-4c8e-8bce-90058a78f502)

## Conclusões 
Portanto, para o problema de detecção de fraude em cartão de crédito analisado neste projeto, o classificador de Machine Learning que apresentou os resultados mais promissores, de acordo com as técnicas analisadas, foi o modelo **Decision Tree**, treinado com o conjunto de dados balanceado utilizando a técnica SMOTE. Os resultados obtidos reforçam a utilidade do modelo **Decision Tree** como uma ferramenta valiosa na detecção de fraudes nesse contexto específico.

## Autores
- Andressa Gomes Moreira - andressagomes@alu.ufc.br
