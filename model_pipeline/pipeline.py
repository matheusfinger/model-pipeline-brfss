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

def load_dataset(dataset_path) -> pd.DataFrame:
    """
    Carrega um arquivo CSV com métricas extraídas e rótulos associados.

    Parameters:
        dataset_path (str): Caminho para o arquivo CSV do dataset.

    Returns:
        sdp_dataset (pandas.DataFrame): DataFrame com os dados carregados.
    """
    return pd.read_csv(dataset_path)

def save_model(model, model_name, data_balance, cv_criteria):
    """Salva o modelo em arquivo pickle"""
    with open(f"model-{model_name}-{cv_criteria.upper()}-{data_balance}.pkl", "wb") as f:
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

def save_metrics(metrics, filename="model_metrics.json"):
    """Salva as métricas do modelo em um arquivo JSON"""
    import json
    with open(filename, 'w') as f:
        # Convertendo numpy arrays e outros tipos para serializáveis
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                serializable_metrics[k] = v.tolist()
            else:
                serializable_metrics[k] = v
        json.dump(serializable_metrics, f, indent=4)

def create_preprocessor():
    """Cria o pré-processador para as colunas do dataset de diabetes"""
    # Colunas para normalização (variáveis contínuas/ordinais)
    columns_to_normalize = ["GenHlth", "PhysHlth", "MentHlth", "PoorHlth", "Educa", "Income2", "Checkup1"]
    # Colunas para pass-through (variáveis binárias/categóricas)
    columns_to_pass_through = ["Sex", "Marital", "Employ1", "HlthPln1", "BpHigh4", "ToldHi2", "CvdStrk3", "ChcScncr", "ChcoCncr", "ChcCopd1", "HavArth3", "AddEpev2", "ChkIdny", "DiffWalk"]

    return ColumnTransformer(transformers=[
          ('num_pipeline', Pipeline([
              ('imputer', SimpleImputer(strategy='mean')),  # Imputação com média
              ('scaler', MinMaxScaler())                    # Normalização Min-Max
          ]), columns_to_normalize),

          ('binary_pipeline', Pipeline([
              ('imputer', SimpleImputer(strategy='most_frequent'))  # Imputação com moda
          ]), columns_to_pass_through)
      ],
      remainder='passthrough'  # Qualquer coluna não listada será passada sem alteração
    )

def create_balance_pipeline(model, data_balance):
    """Cria o pipeline com a técnica de balanceamento especificada"""
    if data_balance == 'SMOTE':
        return ImbPipeline([
            ('preprocessor', create_preprocessor()),
            ('sampler', SMOTE(random_state=42)),
            ('classifier', model)
        ])
    elif data_balance == 'ADASYN':
        return ImbPipeline([
            ('preprocessor', create_preprocessor()),
            ('sampler', ADASYN(random_state=42)),
            ('classifier', model)
        ])
    elif data_balance == 'Undersampling':
        return ImbPipeline([
            ('preprocessor', create_preprocessor()),
            ('sampler', RandomUnderSampler(random_state=42)),
            ('classifier', model)
        ])
    elif data_balance == 'SMOTE+Undersampling':
        return ImbPipeline([
            ('preprocessor', create_preprocessor()),
            ('sampler', SMOTE(random_state=42)),
            ('undersampler', RandomUnderSampler(random_state=42)),
            ('classifier', model)
        ])
    elif data_balance == 'ADASYN+Undersampling':
        return ImbPipeline([
            ('preprocessor', create_preprocessor()),
            ('sampler', ADASYN(random_state=42)),
            ('undersampler', RandomUnderSampler(random_state=42)),
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
                grid_params_list[model_name],
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


def build_champion_model(dataset, x_features, y_label, data_balance, model_info, cv_criteria):
    """Constrói o modelo campeão"""
    X = dataset[x_features]
    y = dataset[y_label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cria pipeline
    pipeline = create_balance_pipeline(model_info["instance"], data_balance)

    # Treina com GridSearch
    grid_model = GridSearchCV(
        pipeline,
        model_info["grid_params"],
        cv=5,
        scoring=cv_criteria,
        n_jobs=-1
    )

    # Treinando modelo campeão
    grid_model.fit(X_train, y_train)
    y_pred = grid_model.predict(X_test)
    y_prob = grid_model.predict_proba(X_test)[:, 1] if hasattr(grid_model, "predict_proba") else None

    metrics_scores = extract_model_metrics_scores(y_test, y_pred, y_prob)

    save_model(grid_model.best_estimator_, model_info["name"], data_balance, cv_criteria)

    return metrics_scores

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


def start(dataset_path):
    """Função principal"""
    logging.basicConfig(
        filename='diabetes_model.log', 
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Carrega o dataset
    dataset = load_dataset(dataset_path)
    # remover depois
    dataset = dataset.head(500)
    dataset['Diabetes_binary'] = dataset['Diabetes_binary'].round().astype(int)
    x_features = dataset.columns.drop('Diabetes_binary').tolist()
    y_label = 'Diabetes_binary'
    # Define os modelos e parâmetros
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42)
    }
    
    grid_params_list = {
        "DecisionTree": {
            'classifier__max_depth': [3, 5, 7, None],
            'classifier__criterion': ['gini', 'entropy'],
            'sampler__sampling_strategy': [0.5, 0.7, 1.0]  # Parâmetros para SMOTE
        }
    }

    balance_methods = ["SMOTE", "ADASYN", "Undersampling", "SMOTE+Undersampling", "ADASYN+Undersampling"]
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
            grid_params_list=grid_params_list,
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
        "grid_params": grid_params_list[best_model_name]
    }

    logger.info("Treinando modelo final...")
    metrics = build_champion_model(
        dataset=dataset,
        x_features=x_features,
        y_label=y_label,
        data_balance=best_method,
        model_info=model_info,
        cv_criteria="f1"
    )

    logger.info("Métricas finais:")
    for metric, value in metrics.items():
        if metric not in ['confusion_matrix', 'classification_report']:
            logger.info(f"{metric}: {value:.4f}")

    filename = f"metrics-{best_model_name}-{best_method}-f1.json"
    logger.info(f"Salvando métricas em {filename}...")
    save_metrics(metrics=metrics, filename=filename)

    return {
        'model_name': best_model_name,
        'resampling_method': best_method,
        'metrics': metrics
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        start(dataset_path)
    else:
        print("Por favor, forneça o caminho para o dataset como argumento.")