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

![Apresentação1](https://github.com/andressagomes26/credit_card_fraud_detection/assets/60404990/2927670d-9258-497c-90ef-e16b0aecbc29)

As fases do projeto foram as seguintes: 

- **Plano de Negócio:** Entender o contexto e os objetivos para a resolução do problema
- **Análise exploratória dos dados:** O objetivo da etapa é entender o comportamento dos dados, como as classes estão divididas, entender se existem falhas no dataset, por exemplo, dados duplicados e faltantes. Além, de visualizar alguns comportamentos dos dados, como, a correlação entre as variáves e as distribuições referentes ao tempo e valor.
- **Pré-processamento dos dados:** Não foi necessário realizar uma limpeza profunda nos dados, pois não existiam casos de valores faltantes e duplicados. Todavia, foi necessário, balancear a base de dados, por se tratar de dataset desbalanceado, na qual a maioria das transações representam transações não fraudulentas. Essa etapa foi realizada para que os dados de treinamento dos modelos de Machine Learning estivessem com qualidade.
- **Treinamento dos modelos:**
- **Análise dos resultados:**

<!--
- Também foram feitas transformação de variáveis e normalização dos dados. Tudo isso para que tivéssemos dados de qualidade para aplicar os modelos de machine learning (ML daqui para frente).
Predição de churn e balanceamento dos dados: Por se tratar de um dataset desbalanceado (contendo muito mais clientes que não realizaram churn do que aqueles que realizaram) foi necessário utilizar alguma técnica de resampling. Eu escolhi o SMOTE. Também fiz a predição utilizando vários modelos de ML, tais quais a regressão logística, a árvore de decisões, a floresta aleatória e o XGBoost. A principal métrica de avaliação dos modelos foi o recall.


## Resultados

Para a extração de caracteres utilizando Pytesseract destacou-se as regiões de interesse da imagem, converteu a imagem para tons de cinza, suavizou a imagem e por fim realizou-se a binarização de Otsu. 

<img src="https://github.com/andressagomes26/character-recognition-pdi/blob/main/notebooks/imagens/imagem_resultado_processamento.jpg">

O Pytesseract conseguiu reconhecer bem os nomes dos produtos, entretanto, para os numerais, a técnica não apresentou resultados interessantes. Ademais, para melhorar o resultado, destacou-se apenas o texto desejado. O resultado do Pytesseract para o texto em destaque:

![WhatsApp Image 2023-05-26 at 09 09 13](https://github.com/andressagomes26/character-recognition-pdi/assets/60404990/0fe0e057-d65f-45d9-a0f1-d7ff84bdd80e)

Em seguida, para realizar a extração dos dígitos dos preços dos produtos texto realizou-se o treinamento de um modelo CNN. Foi necessário adaptar as imagens enviadas para rede neural, pois, a rede será foi treinada com a base de dados MNIST. Logo, é interessante que a imagem de teste possua um formato semelhante, ou seja, a área de interesse (numeral) branca e o fundo preto. Assim, a imagem foi transformada para escala de cinza, suavizada, detectou-se as bordas com o filtro de Canny e por fim, aplicou-se as operações morfológicas de dilatação e erosão, resultando na seguinte imagem:

<img src='https://github.com/andressagomes26/character-recognition-pdi/blob/main/notebooks/imagens/img_cnn_erosao.jpg'>

Por fim, a rede CNN exibiu os seguintes resultados para reconhecimento dos dígitos:
 
![WhatsApp Image 2023-05-26 at 09 11 28](https://github.com/andressagomes26/character-recognition-pdi/assets/60404990/8bbfd518-c56c-4f72-81ee-e520f3f61f2d)
-->
## Autores
- Andressa Gomes Moreira - andressagomes@alu.ufc.br
