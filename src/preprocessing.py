"""
Limpeza, pré-processamento e engenharia de features.

Responsabilidade:
    - Substituir zeros impossíveis por NaN
    - Imputar valores ausentes (mediana estratificada)
    - Criar features derivadas (feature engineering)
    - Escalar features numéricas
    - Dividir dataset em treino/teste com estratificação
    - Salvar dataset processado

Princípio Fundamental — Evitar Data Leakage:
    Todo fitting (mediana, scaler) deve ocorrer APENAS no conjunto de treino.
    O conjunto de teste recebe apenas a transformação, nunca informa os parâmetros.

    Correto:  Fit no treino → Transform no treino e no teste
    Errado:   Fit em todo o dataset → divide → transforma
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.load_data import load_and_validate


# Configurações

PROCESSED_DIR = os.path.join("data", "processed")
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80% treino, 20% teste

# Zeros biologicamente impossíveis → serão convertidos para NaN
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Features finais para modelagem (após engenharia)
FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    # Features de engenharia (adicionadas abaixo)
    "AgeGroup",
    "BMICategory",
    "GlucoseCategory",
    "Insulin_Missing",
    "GlucoseXBMI",
]

TARGET_COL = "Outcome"


# Etapa 1: Tratamento de Zeros


def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte zeros biologicamente impossíveis em NaN.

    Justificativa clínica:
        Glicemia, pressão arterial, espessura cutânea, insulina e IMC
        não podem ser zero em uma pessoa viva. Esses zeros representam
        ausência de medição, não o valor zero.

    Justificativa estatística:
        Tratar zeros como valores reais distorceria:
        - Média e desvio padrão (downward bias)
        - Correlações com a variável alvo
        - Fronteiras de decisão dos modelos

    Parâmetros
    ----------
    df : pd.DataFrame
        Dataset com zeros problemáticos.

    Retorna
    -------
    pd.DataFrame
        Dataset com zeros substituídos por NaN.
    """
    df_clean = df.copy()
    for col in ZERO_AS_MISSING:
        n_zeros = (df_clean[col] == 0).sum()
        df_clean.loc[df_clean[col] == 0, col] = np.nan
        if n_zeros > 0:
            print(f"  [{col}] {n_zeros} zeros convertidos para NaN")

    return df_clean


# Etapa 2: Engenharia de Features


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas com justificativa clínica.

    Features criadas:
        1. AgeGroup: Faixas etárias de risco para DM2
           - Justificativa: ADA recomenda rastreamento diferenciado por faixa
        2. BMICategory: Categorias de IMC (OMS)
           - Justificativa: Risco metabólico é categorizado, não linear
        3. GlucoseCategory: Faixas glicêmicas (ADA 2023)
           - Justificativa: Critério diagnóstico oficial
        4. Insulin_Missing: Flag binário de insulina ausente
           - Justificativa: A ausência em si pode ser informativa
             (médico não solicitou → paciente sem sintomas graves?)
        5. GlucoseXBMI: Interação glicose × IMC
           - Justificativa: Síndrome metabólica combina hiperglicemia
             e obesidade — a interação pode capturar risco sinérgico

    Parâmetros
    ----------
    df : pd.DataFrame
        Dataset com NaN já substituídos.

    Retorna
    -------
    pd.DataFrame
        Dataset com features adicionais.
    """
    df_fe = df.copy()

    # 1. Faixas etárias de risco (ADA guidelines)
    bins_age = [20, 30, 40, 50, 60, 100]
    labels_age = [0, 1, 2, 3, 4]  # Encoded ordinal
    df_fe["AgeGroup"] = pd.cut(
        df_fe["Age"], bins=bins_age, labels=labels_age, right=False
    ).astype(float)

    # 2. Categorias de IMC (OMS)
    # 0=Abaixo do peso, 1=Normal, 2=Sobrepeso, 3=Obesidade I, 4=Obesidade II+
    bins_bmi = [0, 18.5, 25.0, 30.0, 35.0, 100]
    labels_bmi = [0, 1, 2, 3, 4]
    df_fe["BMICategory"] = pd.cut(
        df_fe["BMI"], bins=bins_bmi, labels=labels_bmi, right=False
    ).astype(float)

    # 3. Categorias glicêmicas (ADA 2023, jejum em mg/dL)
    # 0=Normal (<100), 1=Pré-diabetes (100-125), 2=Diabetes (≥126)
    df_fe["GlucoseCategory"] = pd.cut(
        df_fe["Glucose"],
        bins=[0, 100, 126, 300],
        labels=[0, 1, 2],
        right=False,
    ).astype(float)

    # 4. Flag de insulina ausente (dado disponível = 0, ausente = 1)
    df_fe["Insulin_Missing"] = df_fe["Insulin"].isna().astype(int)

    # 5. Interação Glicose × IMC (produto normalizado)
    df_fe["GlucoseXBMI"] = df_fe["Glucose"] * df_fe["BMI"]

    print(f"  Features criadas: AgeGroup, BMICategory, GlucoseCategory, Insulin_Missing, GlucoseXBMI")

    return df_fe


# Etapa 3: Divisão Treino/Teste


def split_data(df: pd.DataFrame):
    """
    Divide o dataset em treino e teste com estratificação.

    Por que estratificar:
        Com 35% de positivos, uma divisão aleatória simples pode
        gerar conjuntos com proporções diferentes. Estratificação
        garante que treino e teste tenham a mesma proporção de classes.

    Por que 80/20:
        Com apenas 768 observações, reservar 20% (154 obs) para teste
        é o equilíbrio razoável entre ter dados suficientes para treinar
        e ter teste representativo.

    Parâmetros
    ----------
    df : pd.DataFrame
        Dataset completo (com features de engenharia, sem imputação ainda).

    Retorna
    -------
    tuple: (X_train, X_test, y_train, y_test)
    """
    # Selecionar apenas colunas que existem no DataFrame
    available_features = [col for col in FEATURE_COLS if col in df.columns]
    X = df[available_features]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # Garante proporção de classes em treino e teste
    )

    print(f"\n  Treino: {X_train.shape[0]} obs ({y_train.mean()*100:.1f}% positivos)")
    print(f"  Teste:  {X_test.shape[0]} obs ({y_test.mean()*100:.1f}% positivos)")

    return X_train, X_test, y_train, y_test


# Etapa 4: Imputação e Escalonamento


def fit_transform_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Aplica imputação e escalonamento sem data leakage.

    Pipeline:
        Treino: fit_transform (aprende parâmetros E transforma)
        Teste:  transform apenas (aplica parâmetros do treino)

    Imputação por mediana:
        Preferida à média porque:
        - Robusta a outliers (um valor extremo não desloca a mediana)
        - Adequada para distribuições assimétricas (como Insulin)
        - Clinicamente mais intuitiva (valor "típico" do paciente)

    StandardScaler (Z-score):
        Transforma cada feature para média=0, desvio=1
        z = (x - μ) / σ
        Necessário para:
        - Logistic Regression (convergência e regularização)
        - KNN (distâncias são sensíveis à escala)
        Inofensivo para modelos baseados em árvores (Random Forest, GBM).

    Parâmetros
    ----------
    X_train, X_test : pd.DataFrame

    Retorna
    -------
    tuple: (X_train_processed, X_test_processed, imputer, scaler)
    """
    # 1. Imputação por mediana — fit APENAS no treino
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)  # Usa medianas do treino

    # 2. Escalonamento — fit APENAS no treino
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)  # Usa média/std do treino

    # Reconverter para DataFrame (mantém nomes das colunas)
    feature_names = X_train.columns.tolist()
    X_train_processed = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

    print(f"\n  Imputação: {imputer.strategy} (mediana de cada feature no treino)")
    print(f"  Escalonamento: StandardScaler (μ=0, σ=1)")
    print(f"  Shape treino processado: {X_train_processed.shape}")
    print(f"  Shape teste processado:  {X_test_processed.shape}")
    print(f"  Valores NaN restantes:   {X_train_processed.isna().sum().sum()}")

    return X_train_processed, X_test_processed, imputer, scaler


# Etapa 5: Salvar Artefatos


def save_processed_data(
    X_train, X_test, y_train, y_test,
    imputer, scaler,
) -> None:
    """
    Salva datasets processados e parâmetros de transformação.

    Estrutura de arquivos:
        data/processed/
            X_train.csv         → Features de treino (processadas)
            X_test.csv          → Features de teste (processadas)
            y_train.csv         → Alvo de treino
            y_test.csv          → Alvo de teste
            imputer_medians.csv → Medianas usadas para imputação (auditabilidade)
            scaler_params.csv   → Parâmetros do scaler (auditabilidade)

    Por que salvar os parâmetros:
        Auditabilidade — você deve conseguir explicar cada transformação
        aplicada ao modelo. Em contexto regulatório (saúde), isso é crítico.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Datasets
    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=True)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=True)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=True)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=True)

    # Parâmetros do imputador
    feature_names = X_train.columns.tolist()
    pd.DataFrame({
        "feature": feature_names,
        "median_imputed": imputer.statistics_,
    }).to_csv(os.path.join(PROCESSED_DIR, "imputer_medians.csv"), index=False)

    # Parâmetros do scaler
    pd.DataFrame({
        "feature": feature_names,
        "mean": scaler.mean_,
        "std": scaler.scale_,
    }).to_csv(os.path.join(PROCESSED_DIR, "scaler_params.csv"), index=False)

    print(f"\n[OK] Dados processados salvos em: {PROCESSED_DIR}/")


# Pipeline Principal


def run_preprocessing(df: pd.DataFrame = None):
    """
    Executa pipeline completo de pré-processamento.

    Fluxo:
        Dados brutos
            → Substituir zeros por NaN
            → Criar features derivadas
            → Dividir treino/teste (estratificado)
            → Imputar (fit no treino)
            → Escalar (fit no treino)
            → Salvar artefatos

    Retorna
    -------
    tuple: (X_train, X_test, y_train, y_test, imputer, scaler)
    """
    print("\n" + "=" * 60)
    print("INICIANDO PRÉ-PROCESSAMENTO")
    print("=" * 60)

    if df is None:
        df = load_and_validate()

    print("\n[1/5] Substituindo zeros impossíveis por NaN...")
    df_clean = replace_zeros_with_nan(df)

    print("\n[2/5] Criando features derivadas...")
    df_engineered = create_features(df_clean)

    print("\n[3/5] Dividindo treino/teste (estratificado 80/20)...")
    X_train, X_test, y_train, y_test = split_data(df_engineered)

    print("\n[4/5] Imputando e escalando (sem data leakage)...")
    X_train_proc, X_test_proc, imputer, scaler = fit_transform_pipeline(X_train, X_test)

    print("\n[5/5] Salvando artefatos...")
    save_processed_data(X_train_proc, X_test_proc, y_train, y_test, imputer, scaler)

    print("\n[OK] Pré-processamento concluído com sucesso.")
    return X_train_proc, X_test_proc, y_train, y_test, imputer, scaler


# Execução direta

if __name__ == "__main__":
    run_preprocessing()