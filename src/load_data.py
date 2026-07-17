"""
Módulo de ingestão e validação inicial dos dados.
    - Carregar o dataset
    - Validar estrutura (colunas, tipos, shape)
    - Retornar DataFrame
"""

import os
import pandas as pd
import numpy as np


RAW_DATA_PATH = os.path.join("data", "raw", "pima_indians_diabetes.csv")

EXPECTED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

EXPECTED_DTYPES = {
    "Pregnancies": "int64",
    "Glucose": "int64",
    "BloodPressure": "int64",
    "SkinThickness": "int64",
    "Insulin": "int64",
    "BMI": "float64",
    "DiabetesPedigreeFunction": "float64",
    "Age": "int64",
    "Outcome": "int64",
}

EXPECTED_SHAPE = (768, 9)


#  Funções


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Carrega o dataset
    Parâmetros
    ----------
    filepath : str
        Caminho para o arquivo CSV. Default: data/raw/pima_indians_diabetes.csv

    Retorna
    -------
    pd.DataFrame
        DataFrame com os dados brutos carregados.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não for encontrado no caminho especificado.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[ERRO] Arquivo não encontrado: '{filepath}'\n"
            f"Verifique se o CSV está em: data/raw/pima_indians_diabetes.csv\n"
            f"Download: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"
        )

    df = pd.read_csv(filepath)
    print(f"[OK] Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Valida se possui a estrutura esperada.

    Verificações realizadas:
        1. Colunas presentes e na ordem correta
        2. Shape esperado (768 × 9)
        3. Tipos de dados compatíveis
        4. Variável alvo contém apenas valores 0 e 1

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser validado.

    Retorna
    -------
    bool
        True se todas as validações passarem.

    Raises
    ------
    ValueError
        Se qualquer validação falhar.
    """
    errors = []

    # 1. Colunas
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)

    if missing_cols:
        errors.append(f"Colunas ausentes: {missing_cols}")
    if extra_cols:
        errors.append(f"Colunas inesperadas: {extra_cols}")

    # 2. Shape
    if df.shape != EXPECTED_SHAPE:
        errors.append(
            f"Shape inesperado: encontrado {df.shape}, esperado {EXPECTED_SHAPE}"
        )

    # 3. Variável alvo
    if "Outcome" in df.columns:
        unique_outcomes = set(df["Outcome"].unique())
        if not unique_outcomes.issubset({0, 1}):
            errors.append(
                f"Variável alvo 'Outcome' contém valores inesperados: {unique_outcomes}"
            )

    # 4. Reportar ou aprovar
    if errors:
        error_msg = "\n".join([f"  - {e}" for e in errors])
        raise ValueError(f"\n[ERRO] Validação do schema falhou:\n{error_msg}")

    print("[OK] Schema validado com sucesso.")
    return True


def get_basic_info(df: pd.DataFrame) -> None:
    """
    Exibe informações básicas no console.

    Apenas imprime para o terminal.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser inspecionado.
    """
    print("\n" + "=" * 60)
    print("INFORMAÇÕES BÁSICAS DO DATASET")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} observações × {df.shape[1]} features")

    print("\nTipos de dados:")
    print(df.dtypes.to_string())

    print("\nDistribuição da variável alvo (Outcome):")
    outcome_counts = df["Outcome"].value_counts()
    outcome_pct = df["Outcome"].value_counts(normalize=True) * 100
    summary = pd.DataFrame({
        "Contagem": outcome_counts,
        "Percentual (%)": outcome_pct.round(1),
    })
    summary.index = summary.index.map({0: "Negativo (0)", 1: "Positivo (1)"})
    print(summary.to_string())

    print("\nPrimeiras 5 linhas:")
    print(df.head().to_string())

    print("\nEstatísticas descritivas:")
    print(df.describe().round(2).to_string())
    print("=" * 60)


def load_and_validate(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Carrega, valida e exibe informações do dataset.

    Esta é a função principal a ser chamada pelos demais módulos.
    Encapsula load_raw_data + validate_schema + get_basic_info.

    Parâmetros
    ----------
    filepath : str
        Caminho para o arquivo CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame validado e pronto para análise.

    Exemplo
    -------
    >>> from src.load_data import load_and_validate
    >>> df = load_and_validate()
    """
    df = load_raw_data(filepath)
    validate_schema(df)
    get_basic_info(df)
    return df


# ─── Execução direta ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_validate()
    print("\n[INFO] Dataset disponível na variável 'df'")
    print(f"[INFO] Memória utilizada: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")