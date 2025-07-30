import pandas as pd
import pickle
import logging
import sys
import numpy as np
from collections import Counter
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score
import requests
import os
import json
from datetime import datetime

def download_dataset() -> pd.DataFrame:
    """
    Faz o download do dataset mais recente.

    Returns:
        sdp_dataset (pandas.DataFrame): DataFrame com os dados carregados.
    """
    ultimo_ano_valido = checa_ultimo_dataset()
    
    if ultimo_ano_valido is not None:
        url_download = 'https://raw.githubusercontent.com/adriabarreto/pipeline-brfss/main/data/cleaned/brfss_cleaned_{ultimo_ano_valido}.csv'
        nome_arquivo = f'brfss_cleaned_{ultimo_ano_valido}.csv'
        
        try:
            response = requests.get(url_download, timeout=30)
            response.raise_for_status()  # Verifica erros
            
            with open(nome_arquivo, 'wb') as f:
                f.write(response.content)
            
            return (nome_arquivo, ultimo_ano_valido)
        except requests.exceptions.RequestException as e:
            return None
    else:
        return None
    
def checa_ultimo_dataset() -> int:
    base_url = 'https://raw.githubusercontent.com/adriabarreto/pipeline-brfss/main/data/cleaned/brfss_cleaned_{ano}.csv'
    ano = 2016  # Começa em 2016
    ano_atual = int(datetime.now().year)
    ultimo_ano_valido = None

    while ano <= ano_atual:
        url = base_url.format(ano=ano)
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            
            if response.status_code == 200:
                ultimo_ano_valido = ano
                ano += 1  # Vai para o próximo ano
            elif response.status_code == 404:
                ano += 1
                continue
            else:
                ano += 1
                continue
                
        except requests.exceptions.RequestException as e:
            ano += 1
            continue
        ano += 1
    return ultimo_ano_valido

def load_dataset(dataset_path) -> pd.DataFrame:
    """
    Carrega um arquivo CSV com os dados.

    Parameters:
        dataset_path (str): Caminho para o arquivo CSV do dataset.

    Returns:
        sdp_dataset (pandas.DataFrame): DataFrame com os dados carregados.
    """
    return pd.read_csv(dataset_path)

def save_model(model):
    """Salva o modelo em arquivo pickle"""
    with open(f"best_model.pkl", "wb") as f:
        pickle.dump(model, f)

def load_model(file_model_path):
    """
    Salva o modelo fornecido em um arquivo pickle.

    Parameters:
        model (obj): Instância do modelo a ser salva.
        model_name (str): Nome identificador do modelo.
        data_balance (str): Método de balanceamento de dados utilizado.
        cv_criteria (str): Critério de validação cruzada utilizado.
    """
    return pickle.load(open(file_model_path, 'rb'))

def extract_model_metrics_scores(y_test, y_pred, y_prob=None) -> dict:
    """Extrai métricas de avaliação"""
    scores = {
        "accuracy_score": metrics.accuracy_score(y_test, y_pred),
        "roc_auc_score": metrics.roc_auc_score(y_test, y_pred) if y_prob is not None else None,
        "f1_score": metrics.f1_score(y_test, y_pred),
        "precision_score": metrics.precision_score(y_test, y_pred),
        "recall_score": metrics.recall_score(y_test, y_pred),
        "confusion_matrix": metrics.confusion_matrix(y_test, y_pred),
        "classification_report": metrics.classification_report(y_test, y_pred)
    }
    return scores

def save_metrics(metrics, model_info, ano, filename="model_metrics.json"):
    """
    Salva métricas e informações do modelo em um arquivo JSON.

    Parâmetros:
    -----------
    metrics : dict
        Dicionário com métricas de avaliação (accuracy, precision, etc.)
    model_info : dict
        Dicionário com informações do modelo contendo:
        - model_params: hiperparâmetros do modelo
        - resampling_method: método de reamostragem (opcional)
        - resampling_params: parâmetros de reamostragem (opcional)
        - dataset_date: data do dataset (opcional)
    filename : str, opcional
        Nome do arquivo de saída (padrão: 'model_metrics.json')
    """
        
    output_data = {
        "metrics": {},
        "model_info": model_info,
        "ano": ano
    }
    
    # Converte métricas para tipos serializáveis
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            output_data["metrics"][k] = v.tolist()
        elif hasattr(v, 'item'):  # Converte numpy scalars
            output_data["metrics"][k] = v.item()
        else:
            output_data["metrics"][k] = v
    
    # Salva em JSON
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=4)

def create_preprocessor():
    """Cria o pré-processador para as colunas do dataset de diabetes"""
     # Colunas para normalização (contínuas/ordinais)
    columns_to_normalize = [
        "BMI", 
        "GenHlth", 
        "PhysHlth", 
        "MentHlth", 
        "Education", 
        "Income", 
        "PhysActivity"
    ]
    
    # Colunas para pass-through (binárias/categóricas)
    columns_to_pass_through = [
        "Sex", 
        "HighBP", 
        "CholCheck", 
        "Smoker", 
        "HvyAlcoholConsump", 
        "AnyHealthcare", 
        "NoDocbcCost", 
        "HeartDiseaseorAttack", 
        "Stroke", 
        "DiffWalk", 
        "HighChol", 
        "Fruits", 
        "Veggies"
    ]

    return ColumnTransformer(
        transformers=[
            # Pipeline para normalização
            ('num_pipeline', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),  # Imputação com média
                ('scaler', MinMaxScaler())                    # Normalização Min-Max
            ]), columns_to_normalize),

            # Pipeline para variáveis binárias/categóricas
            ('binary_pipeline', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent'))  # Imputação com moda
            ]), columns_to_pass_through)
        ],
        remainder='passthrough'  # Mantém colunas não listadas (ex: 'Age' ou 'Diabetes_binary')
    )

def create_balance_pipeline(model, data_balance):
    """Cria o pipeline com a técnica de balanceamento especificada"""
    if data_balance == 'SMOTE':
        return ImbPipeline([
            ('preprocessor', create_preprocessor()),
            ('sampler', SMOTE(random_state=42)),  # Nome do passo: 'sampler'
            ('classifier', model)                  # Nome do passo: 'classifier'
        ])
    elif data_balance == 'ADASYN':
        return ImbPipeline([
            ('preprocessor', create_preprocessor()),
            ('sampler', ADASYN(random_state=42)),  # Nome do passo: 'sampler'
            ('classifier', model)
        ])
    elif data_balance == 'Undersampling':
        return ImbPipeline([
            ('preprocessor', create_preprocessor()),
            ('sampler', RandomUnderSampler(random_state=42)),  # Nome do passo: 'sampler'
            ('classifier', model)
        ])
    else:
        return Pipeline([
            ('preprocessor', create_preprocessor()),
            ('classifier', model)
        ])

def run_experiment(dataset, x_features, y_label, data_balance, models, grid_params_list, cv_criteria) -> dict:
    """Executa experimentos com validação cruzada"""
    X = dataset[x_features]
    y = dataset[y_label]

    skf = StratifiedKFold(n_splits=5)
    models_info_per_fold = {}

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        models_info = {}
        for model_name, model in models.items():
            # Cria pipeline com balanceamento
            pipeline = create_balance_pipeline(model, data_balance)

            # Configura GridSearch
            grid_model = GridSearchCV(
                pipeline,
                grid_params_list,
                cv=3,
                scoring=cv_criteria,
                n_jobs=-1
            )

            # Faz o treinamento
            grid_model.fit(X_train, y_train)

            # Predições com o teste
            y_pred = grid_model.predict(X_test)
            y_prob = grid_model.predict_proba(X_test)[:, 1] if hasattr(grid_model, "predict_proba") else None

            metrics_scores = extract_model_metrics_scores(y_test, y_pred, y_prob)

            models_info[model_name] = {
                "score": metrics_scores,
                "best_estimator": grid_model.best_estimator_,
                "best_params": grid_model.best_params_
            }

        models_info_per_fold[i] = models_info

    return models_info_per_fold


def build_champion_model(dataset, x_features, y_label, data_balance, model_info, cv_criteria, grid_params_list):
    """Constrói o modelo campeão"""
    X = dataset[x_features]
    y = dataset[y_label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cria pipeline
    pipeline = create_balance_pipeline(model_info["instance"], data_balance)

    # Treina com GridSearch
    grid_model = GridSearchCV(
        pipeline,
        grid_params_list[data_balance],
        cv=5,
        scoring=cv_criteria,
        n_jobs=-1
    )

    # Treinando modelo campeão
    grid_model.fit(X_train, y_train)
    y_pred = grid_model.predict(X_test)
    y_prob = grid_model.predict_proba(X_test)[:, 1] if hasattr(grid_model, "predict_proba") else None

    metrics_scores = extract_model_metrics_scores(y_test, y_pred, y_prob)

    save_model(grid_model.best_estimator_)

    return metrics_scores, grid_model.best_params_

def select_best_model(fold_results):
    """Seleciona o melhor modelo baseado na média do F1-score"""
    model_scores = {}

    for fold, models_info in fold_results.items():
        for model_name, info in models_info.items():
            if model_name not in model_scores:
                model_scores[model_name] = []
            model_scores[model_name].append(info["score"]["f1_score"])

    avg_scores = {k: np.mean(v) for k, v in model_scores.items()}
    return max(avg_scores.items(), key=lambda x: x[1])[0]


def train(logger):
    """Função de treino do modelo"""
    teste = download_dataset()
    if teste is None:
        logger.info("Erro ao baixar dataset")
        return None
    dataset_path, ultimo_ano = teste
    # Carrega o dataset
    dataset = load_dataset(dataset_path)
    dataset = dataset
    dataset['Diabetes_binary'] = dataset['Diabetes_binary'].round().astype(int)
    x_features = dataset.columns.drop('Diabetes_binary').tolist()
    y_label = 'Diabetes_binary'
    # Define os modelos e parâmetros
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42)
    }
    
    grid_params_list = {
        "SMOTE": {
            'classifier__max_depth': [5, 7, 10, None],
            'classifier__criterion': ['gini', 'entropy'],
            'sampler__k_neighbors': [3, 5, 7, 10], 
            'sampler__sampling_strategy': [0.5, 0.7, 0.9] 
        },
        "ADASYN": {
            'classifier__max_depth': [5, 7, 10, None],
            'classifier__criterion': ['gini', 'entropy'],
            'sampler__n_neighbors': [3, 5, 7, 10], 
            'sampler__sampling_strategy': [0.5, 0.7, 0.9] 
        },
        "Undersampling": {
            'classifier__max_depth': [5, 7, 10, None],
            'classifier__criterion': ['gini', 'entropy'],
            'sampler__sampling_strategy': [0.5, 0.7, 0.9] 
        }
    }

    balance_methods = ["SMOTE", "ADASYN", "Undersampling"]
    all_results = {}

    logger.info("Iniciando experimentos com diferentes métodos de balanceamento...")
    for method in balance_methods:
        logger.info(f"Rodando experimento com {method}")
        fold_results = run_experiment(
            dataset=dataset,
            x_features=x_features,
            y_label=y_label,
            data_balance=method,
            models=models,
            grid_params_list=grid_params_list[method],
            cv_criteria="f1"
        )
        all_results[method] = fold_results

    # Seleciona melhor combinação modelo + balanceamento
    best_score = -1
    best_model_name = None
    best_method = None

    for method, fold_results in all_results.items():
        avg_scores = {}
        for model_name in models:
            f1s = [fold[model_name]["score"]["f1_score"] for fold in fold_results.values()]
            avg_scores[model_name] = np.mean(f1s)
        top_model = max(avg_scores.items(), key=lambda x: x[1])
        if top_model[1] > best_score:
            best_score = top_model[1]
            best_model_name = top_model[0]
            best_method = method

    logger.info(f"Melhor modelo: {best_model_name} com {best_method} (F1: {best_score:.4f})")

    model_info = {
        "name": best_model_name,
        "instance": models[best_model_name],
        "grid_params": grid_params_list[best_method]
    }

    logger.info("Treinando modelo final...")
    ano = str(ultimo_ano)
    metrics, best_params = build_champion_model(
        dataset=dataset,
        x_features=x_features,
        y_label=y_label,
        data_balance=best_method,
        model_info=model_info,
        cv_criteria="f1",
        grid_params_list=grid_params_list
    )

    logger.info("Métricas finais:")
    for metric, value in metrics.items():
        if metric not in ['confusion_matrix', 'classification_report']:
            logger.info(f"{metric}: {value:.4f}")

    filename = f"model_metrics.json"
    logger.info(f"Salvando métricas em {filename}...")
    infos_modelo = {
        "model_name": best_model_name,
        "model_params": best_params,
        "resampling_method": best_method,
    }
    save_metrics(metrics=metrics, model_info=infos_modelo, ano=ano, filename=filename)

    os.remove(dataset_path)

    return {
        'model_name': best_model_name,
        'resampling_method': best_method,
        'metrics': metrics
    }

def checaModelo() -> list:
    '''
    Retorna lista de modelos salvos no diretório atual
    '''
    arquivos_correspondentes = [arquivo for arquivo in os.listdir() 
                           if arquivo.endswith('.pkl')]

    return arquivos_correspondentes

def testa_modelo(dataset_path: str, model_path: str) -> float:
    '''
    Testa o modelo com os dados do dataset e retorna o F1-score
    
    Parameters:
        dataset_path (str): Caminho para o arquivo CSV do dataset
        model_path (str): Caminho para o arquivo do modelo pickle
    
    Returns:
        float: Valor do F1-score obtido
    '''
    # Carrega o dataset
    dataset = load_dataset(dataset_path)
    dataset['Diabetes_binary'] = dataset['Diabetes_binary'].round().astype(int)
    x_features = dataset.columns.drop('Diabetes_binary').tolist()
    y_label = 'Diabetes_binary'
    
    # Carrega o modelo
    model = load_model(model_path)
    
    # Prepara os dados
    X = dataset[x_features]
    y = dataset[y_label]
    
    # Faz as predições
    y_pred = model.predict(X)
    
    # Calcula o F1-score
    f1 = metrics.f1_score(y, y_pred)
    
    return f1

if __name__ == "__main__":
    logging.basicConfig(
        filename='diabetes_model.log', 
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    arq_modelo = checaModelo()
    if not arq_modelo:
        train(logger)
    elif len(arq_modelo) > 1:
        logger.info("Foram identificados mais de um modelo. Por favor, mantenha somente um modelo no projeto.")
    else:
        caminho_modelo = arq_modelo[0]
        # Abrir e carregar o conteúdo do arquivo JSON
        with open('model_metrics.json', 'r') as f:
            dados = json.load(f)

        # Acessar o valor do ano
        ano_modelo = dados['ano']
        ultimo_ano = checa_ultimo_dataset()
        if ultimo_ano is None:
            logger.error("Falha ao encontrar o dataset mais recente")
            sys.exit(1)
        # Verifica se é atual
        if int(ano_modelo) < ultimo_ano:
            logger.info(f"Modelo existente é de {ano_modelo}, mas dataset mais recente é de {ultimo_ano}")
            logger.info("Testando modelo com dados atuais...")
            try:
                # Baixa o dataset mais recente
                dataset_path, ultimo_ano = download_dataset()
                # Testa se o modelo continua com performance
                f1 = testa_modelo(dataset_path, caminho_modelo)
                logger.info(f"F1-score do modelo atual: {f1:.4f}")
                
                if f1 < 0.3:  # Se performance abaixo do threshold
                    logger.info("Performance abaixo do esperado. Treinando novo modelo...")
                    train(logger)
                else:
                    logger.info("Performance aceitável. Mantendo modelo existente.")
                    # os.remove(dataset_path)
            except Exception as e:
                logger.error(f"Erro ao testar modelo: {str(e)}")
                logger.info("Treinando novo modelo devido ao erro...")
                train(logger)
        else:
            logger.info(f"Modelo existente é o mais atual")
