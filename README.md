<h1> BRFSS Model Pipeline (MLOps) </h1>

Este repositório contém a pipeline de Machine Learning do projeto BRFSS (Behavioral Risk Factor Surveillance System), parte de um trabalho acadêmico da disciplina de Engenharia de Sistemas Inteligentes.

A pipeline automatiza o processo de treinamento, avaliação e exportação do melhor modelo preditivo a partir dos dados limpos gerados pela etapa de DataOps.

<h3> 📁 Estrutura do Projeto </h3>

model-pipeline-brfss/
├── brfss_model/             # Módulo com os scripts
│   └── train.py             # Script principal de treinamento
├── models/                  # Modelo final salvo em binário
│   └── best_model.pkl
├── pyproject.toml           # Configuração do projeto com Poetry
├── README.md
└── .gitignore

<h3> ⚙️ O que esta pipeline faz? </h3>

Carrega os dados limpos da etapa anterior (fora do projeto)

Realiza o pré-processamento para treino

Treina diferentes hiperparâmetros para Árvore de Decisão

Avalia os modelos com validação cruzada e métricas como F1-score, AUC, acurácia

Seleciona o melhor modelo automaticamente

Exporta o modelo final em .pkl para uso posterior

<h3>▶️ Como executar</h3>

Clone o repositório

    git clone https://github.com/matheusfinger/model-pipeline-brfss.git
    cd model-pipeline-brfss

Instale o Poetry (caso ainda não tenha)

    pip install poetry

Instale as dependências

    poetry install

Execute a pipeline

    poetry run python pipeline.py
