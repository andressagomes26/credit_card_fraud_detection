<h1 align="center">Detecção de Fraude de Cartão de Crédito</h1>

## Problema de Negócio

**Contexto:** Um ponto de extrema importância para empresas de cartão de crédito é a capacidade de reconhecer transações fraudulentas para que os clintes não sejam cobrados por itens que não compraram. Sabendo disso, neste projeto serão utilizados algoritimos de Machine Learning para detectar transações de créditos fraudulentas.

## Sobre o conjunto de dados
O conjunto de dados contém transações feitas por cartões de crédito em setembro de 2013 por titulares de cartões europeus.
Este conjunto de dados apresenta transações ocorridas em dois dias, onde temos **492 fraudes em 284.807 transações**. O conjunto de dados é altamente desbalanceado, a classe positiva (fraudes) responde por 0,172% de todas as transações.

As variáveis de entrada **numéricas** são o resultado de uma **transformação PCA**. Devido a questões de confidencialidade, não são fornecidos os recursos originais sobre os dados. As características V1, V2, … V28 são os principais componentes obtidos com PCA, as únicas características que não foram transformadas com PCA são 'Time' e 'Amount'. O recurso 'Time' contém os segundos decorridos entre cada transação e a primeira transação no conjunto de dados. O recurso 'Amount' é o valor da transação. A característica 'Classe' é a variável de resposta e assume **valor 1 em caso de fraude e 0 caso contrário**.

**Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

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
Nesse projeto foi utilizado os modelos KNN e Decision Tree para realizar a detecção de fraudes de cartão de crédito, a partir da análise da base de dados disponível no Kaggle. Para realizar o projeto foi realizada algumas etapas, sendo elas:

![apresentacao](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/3ec20637-d24a-4e10-81e6-0968e55abbc7)

As fases do projeto foram as seguintes: 

- **Plano de Negócio:** Entender o contexto e os objetivos para a resolução do problema.
- **Análise exploratória dos dados:** O objetivo da etapa é entender o comportamento dos dados, como as classes estão divididas, entender se existem falhas no dataset, por exemplo, dados duplicados e faltantes. Além, de visualizar alguns comportamentos dos dados, como, a correlação entre as variáves e as distribuições referentes ao tempo e valor.
- **Pré-processamento dos dados:** Não foi necessário realizar uma limpeza profunda nos dados, pois não existiam casos de valores faltantes e duplicados. Todavia, foi necessário, balancear a base de dados, por se tratar de dataset desbalanceado, na qual a maioria das transações representam transações não fraudulentas. Para isso, utilizou-se as técnicas RandomUnderSampling (RUS) e SMOTE. Essa etapa foi realizada para que os dados de treinamento dos modelos de Machine Learning estivessem com qualidade.
- **Treinamento dos modelos:** Foi realizado o treinamento dos modelos Decision Tree e KNN, que servirão como máquinas preditivas para esse problema. Dessa forma, usou-se a validação cruzada K-fold para avaliar o desempenho dos modelos e estimar a capacidade de generalização em dados não vistos.
- **Análise dos resultados:** Por fim, para analisar os resultados obtidos usou-se as métricas Acurácia, Precisão, Recall e F1-Score, como também, a plotagem da matriz de confusão, Curva ROC e a área sob a curva (AUC), medidas importantes em problemas de classificação binária.

## Análise Exploratória dos Dados
Nessa etapa foi realizada uma análise exploratória para examinar e estudar as características do conjunto de dados. A priori, verificou-se as informações estatísticas dos valores e se o dataset possui valores ausentes ou duplicados. 

Em seguida, foi traçado alguns gráficos para entender o comportamento das variáveis. Em relação a distribuição dos dados em cada classe, é possível concluir que se trata de conjunto de dados distorcido, visto que, existem 284315 amostras para transações não fraudulentas e apenas 492 para as transações fraudulentas

![download](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/e92e01ea-bd52-4450-88be-d13d3c5869d6)

Além disso, foi realizada uma análise das associações entre as variáveis numéricas por meio de uma matriz de correlação. Essa análise permite observar o grau de correlação entre as variáveis. É possível notar que os atributos V2 e Amount apresentam uma correlação negativa, indicando uma relação inversa entre eles. Por outro lado, os atributos V7 e Amount possuem uma correlação positiva, o que sugere uma relação direta entre eles. Outro fato importante é que as variáveis não apresentam alta correlação entre si, visto que, a correlação forte entre as variáveis pode trazer desafios no treinamento dos modelos, podendo fornecer informações redundantes e resultar em overfitting, influenciando diretamente no desempenho do modelo.

![corr](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/aeafb528-919d-4558-9853-6c5ef1d2ac24)

Por fim, realizou-se uma análise das distribuições das variáveis "Tempo" e "Valor" em relação aos grupos de transações normais e fraudulentas. Observa-se que as transações fraudulentas têm uma concentração maior de valores entre 0€ e 1.000€, enquanto as transações normais estão distribuídas entre 0€ e 5.000€. Quanto ao atributo "Tempo", não se observa diferença perceptível entre os dois tipos de transações.

![amout](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/d1495dd2-2d06-436c-bb82-d398f85fbcbb)

## Pré-processamento dos Dados
Nesta etapa, foi realizado o balanceamento dos dados para garantir a qualidade dos dados de treinamento do modelo. Utilizaram-se duas técnicas: RandomUnderSampling (RUS) e SMOTE. A técnica RUS reduz a quantidade de exemplos da classe majoritária, selecionando aleatoriamente uma amostra desses exemplos. Isso resulta em um conjunto de dados balanceado, porém com uma quantidade reduzida de dados. Já a técnica SMOTE gera exemplos sintéticos para a classe minoritária. O SMOTE é especialmente útil quando a classe minoritária apresenta regiões com poucos exemplos, permitindo preencher essas regiões com exemplos sintéticos.

Dessa forma, ao utilizar a técnica RUS, houve uma redução na quantidade de dados de treinamento, sendo utilizados 369 exemplos para a classe de transações não fraudulentas e 369 exemplos para as transações fraudulentas.

![grafico_rus](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/31ab23a5-5858-4604-a57e-458f3dc61298)

Em contrapartida, com técnica SMOTE, foram gerados dados sintéticos de treinamento, totalizando 213.236 amostras para a classe de transações não fraudulentas e 213.236 para as transações fraudulentas.

![grafico_Smote](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/1029b6b1-541b-4c47-ad76-23ee952e1d62)

## Treinamento dos modelos

Durante o treinamento dos modelos Decision Tree e KNN, foi utilizado a técnica GridSearchCV, que auxilia na determinação de valores adequados para os hiperparâmetros e no controle da complexidade do modelo. Desse modo, foram definidos os seguites hiperparâmetros para cada modelo:

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

A técnica que aprensentou resultados mais satisfatórios para o problema em questão foi o modelo Decision Tree treinado com o cojunto dos dados balanceados com a técnica SMOTE. A Tabela 1 apresenta os resultados para as métricas Acurácia, Precisão, Recall e F1-Score em cada técnica.

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

Portanto, para o problema de detecção de fraude em cartão de crédito, analisado neste projeto, o classificador de Machine Learning que apresentou resultados mais promissores, de acordo com as técnicas analisadas foi o **Decision Tree**.

## Autores
- Andressa Gomes Moreira - andressagomes@alu.ufc.br
