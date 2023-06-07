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

![Apresentação1](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/ec430bfc-168b-480b-a565-719ea58f12a1)

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

Ademais, foi analisado, por meio da matriz de correlação, o nível em que as variáveis numéricas estão associadas. Dessa forma, é possível observar que os atributos V2 e Amout possuem uma correlação negativa, enquanto, V7 e Amount possuem correlação positiva. 



## Resultados

A técnica que aprensentou resultados mais satisfatórios para o problema em questão foi o modelo Decision Tree treinado com o cojunto dos dados balanceados com a técnica SMOTE. A Tabela 1 apresenta os resultados para as métricas Acurácia, Precisão, Recall e F1-Score em cada técnica.

Exemplo   | Acurácia | Precisão | Recall | F1-Score | AUC
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
