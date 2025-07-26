<h1> BRFSS Model Pipeline (MLOps) </h1>

Este repositório contém a pipeline de Machine Learning do projeto BRFSS (Behavioral Risk Factor Surveillance System), parte de um trabalho acadêmico da disciplina de Engenharia de Sistemas Inteligentes.

A pipeline automatiza o processo de treinamento, avaliação e exportação do melhor modelo preditivo a partir dos dados limpos gerados pela etapa de DataOps.

<h3> 📁 Estrutura do Projeto </h3>

     model-pipeline-brfss/ <br>
    ├── model_pipeline/             # Módulo com os scripts <br>
    │   └── __init__.py
    │   └── pipeline.py             # Script de treinamento <br>
    ├── model-DecisionTree*.pkl <br> 
    ├── pyproject.toml           # Configuração do projeto com Poetry <br>
    ├── README.md <br>
    └── .gitignore <br>

<h3> ⚙️ O que esta pipeline faz? </h3>
<ul>
<li>Carrega os dados limpos da etapa anterior (fora do projeto)</li>

<li>Aplica diferentes estratégias de balanceamento de classes, como:</li>
<ul>
     <li>RandomUnderSampler</li>
     <li>SMOTE</li>
     <li>ADASYN</li>
     <li>SMOTE+RandomUnderSampler</li>
     <li>ADASYN+RandomUnderSampler</li>
</ul>

<li>Para cada estratégia, treina uma Árvore de Decisão com:</li>
     <ul>
     <li>Otimização de hiperparâmetros via GridSearchCV</li>
     <li>Validação cruzada com métricas como F1-Score</li>
     </ul>

<li>Seleciona a melhor combinação de sampling + árvore otimizada</li>

<li>Exporta o modelo final em .pkl para uso posterior</li>
</ul>
<h3>▶️ Como executar</h3>

Clone o repositório

    git clone https://github.com/matheusfinger/model-pipeline-brfss.git
    cd model-pipeline-brfss

Instale o Poetry (caso ainda não tenha)

    pip install poetry

Instale as dependências

    poetry install

Execute a pipeline

    poetry run python model_pipeline/pipeline.py
