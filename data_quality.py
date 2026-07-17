"""
Relatório de qualidade dos dados.

Responsabilidade:
    - Detectar valores ausentes (explícitos e implícitos)
    - Identificar zeros biologicamente impossíveis
    - Calcular estatísticas de qualidade por feature
    - Detectar duplicatas
    - Gerar relatório em texto

Conceito:
    Zeros em variáveis como Glucose, BloodPressure, BMI são
    biologicamente impossíveis. 
    Representam dados ausentes mal codificados, devem ser tratados como NaN antes de qualquer análise.
"""

import os
import pandas as pd
import numpy as np
from src.load_data import load_and_validate


# Configurações

# Zero biologicamente impossível
ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# Zero é biologicamente válido
ZERO_VALID = [
    "Pregnancies",  # Nunca engravidou → zero válido
]

REPORT_PATH = os.path.join("outputs", "reports", "data_quality_report.txt")


# Funções de Análise


def check_explicit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica valores NaN explícitos no dataset.

    Parâmetros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    pd.DataFrame
        Tabela com contagem e percentual de NaN por coluna.
    """
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

    report = pd.DataFrame({
        "NaN Count": missing_count,
        "NaN %": missing_pct,
    })
    return report[report["NaN Count"] > 0] if report["NaN Count"].sum() > 0 else report


def check_implicit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta zeros biologicamente impossíveis (ausência implícita).

    Estas features não podem ter valor zero em uma pessoa viva:
    - Glucose: glicemia zero = morte
    - BloodPressure: pressão zero = ausência de circulação
    - SkinThickness: dobra cutânea zero = impossível fisicamente
    - Insulin: insulina zero possível mas extremamente raro
    - BMI: IMC zero = impossível

    Parâmetros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    pd.DataFrame
        Tabela com contagem e percentual de zeros problemáticos.
    """
    rows = []
    for col in ZERO_AS_MISSING:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            zero_pct = round(zero_count / len(df) * 100, 2)
            rows.append({
                "Feature": col,
                "Zeros (ausência implícita)": zero_count,
                "% do Total": zero_pct,
                "Severidade": _classify_severity(zero_pct),
            })

    return pd.DataFrame(rows).set_index("Feature")


def _classify_severity(pct: float) -> str:
    """
    Classifica a severidade da taxa de ausência.

    Critério:
        < 5%   → Baixa   (imputação simples é suficiente)
        5–30%  → Média   (avaliar estratégia de imputação)
        > 30%  → Alta    (considerar exclusão ou feature engineering)
    """
    if pct < 5:
        return "🟢 Baixa"
    elif pct <= 30:
        return "🟡 Média"
    else:
        return "🔴 Alta"


def check_duplicates(df: pd.DataFrame) -> dict:
    """
    Verifica duplicatas exatas no dataset.

    Parâmetros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    dict
        Contagem de duplicatas e linhas afetadas.
    """
    n_duplicates = df.duplicated().sum()
    return {
        "total_duplicatas": int(n_duplicates),
        "percentual": round(n_duplicates / len(df) * 100, 2),
        "indices": df[df.duplicated()].index.tolist(),
    }


def check_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta outliers usando o método IQR (Interquartile Range).

    Método IQR:
        - Q1 = percentil 25, Q3 = percentil 75
        - IQR = Q3 - Q1
        - Outlier: valor < Q1 - 1.5*IQR ou > Q3 + 1.5*IQR

    Este método é preferível ao Z-score para dados clínicos porque:
        1. Não assume distribuição normal
        2. É robusto a outliers extremos
        3. Funciona bem com amostras menores

    Parâmetros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    pd.DataFrame
        Tabela com contagem de outliers por feature numérica.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("Outcome")
    rows = []

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        n_outliers = len(outliers)

        rows.append({
            "Feature": col,
            "Q1": round(Q1, 2),
            "Q3": round(Q3, 2),
            "IQR": round(IQR, 2),
            "Limite Inferior": round(lower_bound, 2),
            "Limite Superior": round(upper_bound, 2),
            "N° Outliers": n_outliers,
            "% Outliers": round(n_outliers / len(df) * 100, 2),
        })

    return pd.DataFrame(rows).set_index("Feature")


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estatísticas descritivas estendidas.

    Inclui além do describe() padrão:
        - Assimetria (skewness): valores |skew| > 1 indicam distribuição assimétrica
        - Curtose (kurtosis): valores altos indicam caudas pesadas (mais outliers)
        - Coeficiente de variação (CV): dispersão relativa à média

    Parâmetros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    pd.DataFrame
        Estatísticas descritivas estendidas.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    stats = df[numeric_cols].describe().T
    stats["skewness"] = df[numeric_cols].skew().round(3)
    stats["kurtosis"] = df[numeric_cols].kurtosis().round(3)
    stats["cv_%"] = (df[numeric_cols].std() / df[numeric_cols].mean() * 100).round(2)

    return stats.round(3)


def generate_report(df: pd.DataFrame, save_path: str = REPORT_PATH) -> None:
    """
    Gera relatório completo de qualidade dos dados.

    O relatório é exibido no console e salvo em arquivo texto.

    Parâmetros
    ----------
    df : pd.DataFrame
        Dataset bruto.
    save_path : str
        Caminho para salvar o relatório.
    """
    separator = "=" * 70
    lines = []

    def add(text: str = ""):
        lines.append(text)
        print(text)

    add(separator)
    add("RELATÓRIO DE QUALIDADE DOS DADOS")
    add("Projeto: Diabetes Risk Prediction")
    add(f"Dataset: Pima Indians Diabetes Database")
    add(f"Observações: {df.shape[0]} | Features: {df.shape[1]}")
    add(separator)

    # 1. Ausência explícita
    add("\n1. VALORES AUSENTES EXPLÍCITOS (NaN)")
    add("-" * 40)
    explicit = check_explicit_missing(df)
    if explicit.empty or explicit["NaN Count"].sum() == 0:
        add("Nenhum valor NaN explícito encontrado.")
        add("OBSERVAÇÃO: Zeros problemáticos foram codificados como 0, não NaN.")
    else:
        add(explicit.to_string())

    # 2. Ausência implícita (zeros)
    add("\n2. ZEROS BIOLOGICAMENTE IMPOSSÍVEIS (ausência implícita)")
    add("-" * 40)
    implicit = check_implicit_missing(df)
    add(implicit.to_string())
    add("\nINTERPRETAÇÃO: Estes zeros NÃO representam o valor zero.")
    add("Representam ausência de medição e devem ser tratados como NaN.")

    # 3. Duplicatas
    add("\n3. DUPLICATAS")
    add("-" * 40)
    dupl = check_duplicates(df)
    add(f"Total de linhas duplicadas: {dupl['total_duplicatas']} ({dupl['percentual']}%)")
    if dupl["total_duplicatas"] > 0:
        add(f"Índices: {dupl['indices']}")

    # 4. Outliers
    add("\n4. OUTLIERS (Método IQR)")
    add("-" * 40)
    outliers = check_outliers_iqr(df)
    add(outliers[["N° Outliers", "% Outliers", "Limite Inferior", "Limite Superior"]].to_string())
    add("\nNOTA: Em dados clínicos, outliers podem ser valores reais extremos.")
    add("Investigar caso a caso antes de remover.")

    # 5. Estatísticas descritivas
    add("\n5. ESTATÍSTICAS DESCRITIVAS ESTENDIDAS")
    add("-" * 40)
    stats = compute_descriptive_stats(df)
    add(stats[["mean", "std", "min", "max", "skewness", "kurtosis", "cv_%"]].to_string())
    add("\nINTERPRETAÇÃO DE ASSIMETRIA:")
    add("  |skew| < 0.5  → Distribuição aproximadamente simétrica")
    add("  0.5 ≤ |skew| < 1 → Assimetria moderada")
    add("  |skew| ≥ 1    → Alta assimetria (considerar transformação)")

    # 6. Balanceamento
    add("\n6. BALANCEAMENTO DA VARIÁVEL ALVO")
    add("-" * 40)
    vc = df["Outcome"].value_counts()
    pct = df["Outcome"].value_counts(normalize=True) * 100
    add(f"Classe 0 (Negativo): {vc[0]} obs ({pct[0]:.1f}%)")
    add(f"Classe 1 (Positivo): {vc[1]} obs ({pct[1]:.1f}%)")
    ratio = vc[0] / vc[1]
    add(f"Razão de desbalanceamento: {ratio:.2f}:1")
    if ratio > 3:
        add("⚠️  Desbalanceamento severo — considerar SMOTE ou class_weight='balanced'")
    elif ratio > 1.5:
        add("ℹ️  Desbalanceamento moderado — usar F1-Score e AUC-ROC como métricas")
    else:
        add("✅  Classes aproximadamente balanceadas")

    # 7. Sumário e recomendações
    add("\n7. RECOMENDAÇÕES DE PRÉ-PROCESSAMENTO")
    add("-" * 40)
    add("PRIORIDADE 1 (Crítico):")
    add("  → Substituir zeros em [Glucose, BloodPressure, BMI] por NaN")
    add("  → Imputar com mediana estratificada por Outcome")
    add("")
    add("PRIORIDADE 2 (Importante):")
    add("  → Investigar SkinThickness (29.6% ausente) — mediana ou KNN imputation")
    add("  → Investigar Insulin (48.7% ausente) — criar flag binário de disponibilidade")
    add("")
    add("PRIORIDADE 3 (Opcional):")
    add("  → Avaliar transformação log em Insulin (alta assimetria)")
    add("  → Feature engineering: faixas etárias, categorias de IMC")

    add("\n" + separator)
    add("FIM DO RELATÓRIO")
    add(separator)

    # Salvar relatório
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[OK] Relatório salvo em: {save_path}")


# Execução direta

if __name__ == "__main__":
    df = load_and_validate()
    generate_report(df)