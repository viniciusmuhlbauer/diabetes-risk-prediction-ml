"""
Treinamento e otimização de modelos de classificação.

Responsabilidade:
    - Definir e treinar múltiplos modelos
    - Validação cruzada estratificada (5-fold)
    - Busca de hiperparâmetros com GridSearchCV
    - Serializar modelos treinados

Conceitos:
    Validação cruzada: Divide o treino em K folds, treina em K-1
    e valida no fold restante. Repete K vezes. Resultado: estimativa
    robusta da performance real sem usar o conjunto de teste.

    GridSearchCV: Testa todas as combinações de hiperparâmetros
    via validação cruzada e seleciona a melhor.

    Por que não usar o teste para escolher o modelo:
    Se você escolher o modelo baseado na performance no teste,
    o teste "vaza" informação — ele não é mais independente.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, recall_score


# Configurações

MODELS_DIR = "models"
RANDOM_STATE = 42
CV_FOLDS = 5  # Validação cruzada com 5 folds

# Scoring para GridSearchCV — otimizar por AUC-ROC (mais robusto que accuracy)
SCORING_PRIMARY = "roc_auc"

SCORING_MULTI = {
    "roc_auc": make_scorer(roc_auc_score),
    "f1": make_scorer(f1_score),
    "recall": make_scorer(recall_score),
}


# Definição dos Modelos


def get_model_configs() -> dict:
    """
    Define modelos e grids de hiperparâmetros.

    Justificativas de escolha dos modelos:

    1. Logistic Regression (baseline):
       - Modelo linear interpretável; padrão em estudos clínicos
       - Coeficientes têm interpretação direta (log-odds)
       - C: parâmetro de regularização (inverso de lambda)
         Alto C = menos regularização (pode overfit)
         Baixo C = mais regularização (pode underfit)

    2. Decision Tree:
       - Alta interpretabilidade (pode ser visualizada por médicos)
       - Propensa a overfit sem controle de profundidade
       - max_depth controla complexidade

    3. Random Forest:
       - Ensemble de árvores com bagging
       - Robusto a outliers e dados ausentes residuais
       - n_estimators: mais árvores = menor variância (até certo ponto)

    4. Gradient Boosting:
       - Ensemble sequencial — cada árvore corrige erros da anterior
       - Alta performance, mas mais sensível a hiperparâmetros
       - learning_rate × n_estimators: trade-off fundamental

    5. KNN:
       - Modelo baseado em distância — sensível ao escalonamento
       - Serve como "sanity check" para verificar se o escalonamento
         foi bem aplicado no pré-processamento
       - n_neighbors: mais vizinhos = decisão mais suave (menos overfit)
    """
    configs = {
        "LogisticRegression": {
            "model": LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight="balanced",  # Ajusta para desbalanceamento
            ),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
            },
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            "params": {
                "max_depth": [3, 5, 7, None],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,  # Usar todos os núcleos disponíveis
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5],
            },
        },
        "KNN": {
            "model": KNeighborsClassifier(n_jobs=-1),
            "params": {
                "n_neighbors": [3, 5, 7, 11, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        },
    }
    return configs


# Treinamento com CV


def train_with_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """
    Avalia todos os modelos com validação cruzada estratificada.

    Por que antes do GridSearch:
        Dá uma visão rápida de quais modelos têm potencial,
        com custo computacional baixo (sem busca de hiperparâmetros).

    StratifiedKFold:
        Garante que a proporção de classes seja mantida em cada fold.
        Essencial com dados desbalanceados.

    Parâmetros
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Retorna
    -------
    pd.DataFrame
        Tabela comparativa de performance por modelo.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    configs = get_model_configs()
    results = []

    print(f"\nValidação cruzada ({CV_FOLDS}-fold estratificado):")
    print("-" * 50)

    for name, config in configs.items():
        start = time.time()

        cv_results = cross_validate(
            config["model"],
            X_train, y_train,
            cv=cv,
            scoring=SCORING_MULTI,
            return_train_score=False,
        )

        elapsed = time.time() - start

        results.append({
            "Model": name,
            "AUC-ROC (mean)": cv_results["test_roc_auc"].mean().round(4),
            "AUC-ROC (std)": cv_results["test_roc_auc"].std().round(4),
            "F1 (mean)": cv_results["test_f1"].mean().round(4),
            "Recall (mean)": cv_results["test_recall"].mean().round(4),
            "Tempo (s)": round(elapsed, 2),
        })

        print(
            f"  {name:<25} "
            f"AUC={cv_results['test_roc_auc'].mean():.4f} "
            f"(±{cv_results['test_roc_auc'].std():.4f}) | "
            f"F1={cv_results['test_f1'].mean():.4f} | "
            f"Recall={cv_results['test_recall'].mean():.4f}"
        )

    results_df = pd.DataFrame(results).sort_values("AUC-ROC (mean)", ascending=False)

    print("\nRanking por AUC-ROC:")
    print(results_df.to_string(index=False))

    return results_df


# GridSearch para Melhor Modelo


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = None,
) -> dict:
    """
    Busca de hiperparâmetros via GridSearchCV.

    Se model_name for None, executa para todos os modelos.
    Se especificado, executa apenas para o modelo indicado.

    Parâmetros
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    model_name : str, opcional
        Nome do modelo a otimizar. None = todos.

    Retorna
    -------
    dict
        {nome_modelo: GridSearchCV ajustado}
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    configs = get_model_configs()

    if model_name:
        if model_name not in configs:
            raise ValueError(f"Modelo '{model_name}' não encontrado. Opções: {list(configs.keys())}")
        configs = {model_name: configs[model_name]}

    tuned_models = {}

    print(f"\nOtimização de hiperparâmetros (GridSearchCV, {CV_FOLDS}-fold):")
    print("-" * 60)

    for name, config in configs.items():
        print(f"\n  [{name}] Buscando melhores hiperparâmetros...")
        start = time.time()

        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=cv,
            scoring=SCORING_PRIMARY,
            n_jobs=-1,
            verbose=0,
            refit=True,  # Retreina com melhores params em todo X_train
        )

        grid_search.fit(X_train, y_train)
        elapsed = time.time() - start

        best_score = grid_search.best_score_
        best_params = grid_search.best_params_

        print(f"  Melhor AUC-ROC (CV): {best_score:.4f}")
        print(f"  Melhores parâmetros: {best_params}")
        print(f"  Tempo: {elapsed:.1f}s")

        tuned_models[name] = grid_search

    return tuned_models


# Salvar Modelos


def save_models(tuned_models: dict) -> None:
    """
    Serializa modelos treinados com joblib.

    Por que joblib e não pickle:
        joblib é mais eficiente para objetos NumPy (arrays grandes),
        que é o caso dos estimadores do scikit-learn.

    Parâmetros
    ----------
    tuned_models : dict
        {nome_modelo: GridSearchCV ajustado}
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, grid_search in tuned_models.items():
        # Salva o melhor estimador (já retreinado em todo X_train)
        model_path = os.path.join(MODELS_DIR, f"{name.lower()}_best.pkl")
        joblib.dump(grid_search.best_estimator_, model_path)
        print(f"  [OK] Modelo salvo: {model_path}")

    # Salva também os GridSearchCV completos (para análise de hiperparâmetros)
    for name, grid_search in tuned_models.items():
        gs_path = os.path.join(MODELS_DIR, f"{name.lower()}_gridsearch.pkl")
        joblib.dump(grid_search, gs_path)


def save_cv_results(results_df: pd.DataFrame) -> None:
    """Salva resultados de CV em CSV para análise posterior."""
    path = os.path.join("outputs", "reports", "cv_results.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results_df.to_csv(path, index=False)
    print(f"  [OK] Resultados de CV salvos: {path}")


# Pipeline Principal


def run_training(X_train: pd.DataFrame = None, y_train: pd.Series = None):
    """
    Executa pipeline completo de treinamento.

    Fluxo:
        Dados processados
            → Validação cruzada (todos os modelos)
            → GridSearch (otimização de hiperparâmetros)
            → Salvar modelos e resultados

    Retorna
    -------
    tuple: (tuned_models, cv_results)
    """
    print("\n" + "=" * 60)
    print("INICIANDO TREINAMENTO DOS MODELOS")
    print("=" * 60)

    # Carregar dados processados se não fornecidos
    if X_train is None:
        X_train = pd.read_csv(os.path.join("data", "processed", "X_train.csv"), index_col=0)
        y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv"), index_col=0).squeeze()

    print(f"\nDados de treino: {X_train.shape[0]} obs × {X_train.shape[1]} features")

    print("\n[1/3] Avaliação inicial com validação cruzada...")
    cv_results = train_with_cross_validation(X_train, y_train)
    save_cv_results(cv_results)

    print("\n[2/3] Otimização de hiperparâmetros...")
    tuned_models = tune_hyperparameters(X_train, y_train)

    print("\n[3/3] Salvando modelos...")
    save_models(tuned_models)

    print("\n[OK] Treinamento concluído.")
    return tuned_models, cv_results


# Execução direta

if __name__ == "__main__":
    run_training()