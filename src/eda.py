"""
Análise Exploratória de Dados (EDA).

Responsabilidade:
    - Visualizar distribuições univariadas
    - Analisar correlações entre features
    - Comparar distribuições por classe (Outcome 0 vs 1)
    - Gerar e salvar todas as figuras em outputs/figures/

Perguntas que esta análise responde:
    1. Como cada feature se distribui?
    2. Quais features diferenciam pacientes diabéticos de não-diabéticos?
    3. Existe multicolinearidade entre features?
    4. Como os dados ausentes se distribuem entre as classes?
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from src.load_data import load_and_validate


# Configurações

FIGURES_DIR = os.path.join("outputs", "figures")
RANDOM_STATE = 42

# Paleta clínica: vermelho para positivo, azul para negativo
PALETTE = {0: "#2196F3", 1: "#E53935"}
LABEL_MAP = {0: "Negativo (0)", 1: "Positivo (1)"}

# Estilo visual profissional
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age",
]


# Funções de Visualização


def _save_figure(filename: str) -> None:
    """Salva figura no diretório de outputs."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[OK] Figura salva: {path}")
    plt.close()


def plot_class_distribution(df: pd.DataFrame) -> None:
    """
    Visualiza a distribuição da variável alvo.

    Por que importa: Identifica desbalanceamento de classes.
    Desbalanceamento afeta a escolha de métricas e estratégias de modelagem.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Distribuição da Variável Alvo (Outcome)", fontsize=14, fontweight="bold")

    counts = df["Outcome"].value_counts().sort_index()
    colors = [PALETTE[0], PALETTE[1]]
    labels = ["Negativo (0)", "Positivo (1)"]

    # Gráfico de barras
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Contagem por Classe")
    axes[0].set_ylabel("Número de Observações")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    # Gráfico de pizza
    pcts = df["Outcome"].value_counts(normalize=True) * 100
    axes[1].pie(
        pcts.values,
        labels=[f"{l}\n{p:.1f}%" for l, p in zip(labels, pcts.values)],
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Proporção por Classe")

    plt.tight_layout()
    _save_figure("01_class_distribution.png")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """
    Histogramas com curva KDE para cada feature, separados por classe.

    Por que importa: Permite identificar visualmente quais features
    discriminam melhor as classes (Outcome 0 vs 1).

    KDE (Kernel Density Estimation): Estimativa suave da distribuição
    de probabilidade, sem assumir normalidade.
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(
        "Distribuição das Features por Classe\n(Azul = Negativo | Vermelho = Positivo)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]
        for outcome, color in PALETTE.items():
            subset = df[df["Outcome"] == outcome][col]
            # Remove zeros biologicamente impossíveis para visualização mais honesta
            if col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
                subset = subset[subset > 0]

            ax.hist(
                subset,
                bins=25,
                color=color,
                alpha=0.5,
                label=LABEL_MAP[outcome],
                density=True,
            )
            # KDE sobreposta
            subset.plot.kde(ax=ax, color=color, linewidth=2)

        ax.set_title(col)
        ax.set_xlabel("Valor")
        ax.set_ylabel("Densidade")
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save_figure("02_feature_distributions.png")


def plot_boxplots_by_class(df: pd.DataFrame) -> None:
    """
    Boxplots por classe para cada feature.

    Por que importa:
        - Mostra mediana, IQR e outliers por grupo
        - Identifica diferenças na distribuição entre classes
        - Outliers aparecem como pontos individuais

    Interpretação do boxplot:
        - Linha central: mediana (Q2)
        - Caixa: IQR (Q1 a Q3 — 50% central dos dados)
        - Whiskers: 1.5 × IQR
        - Pontos: outliers
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(
        "Boxplots por Classe\n(Identifica separabilidade entre Diabéticos e Não-Diabéticos)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]

        data_by_class = [
            df[df["Outcome"] == 0][col].values,
            df[df["Outcome"] == 1][col].values,
        ]

        bp = ax.boxplot(
            data_by_class,
            patch_artist=True,
            tick_labels=["Negativo (0)", "Positivo (1)"],
            medianprops={"color": "black", "linewidth": 2},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )

        bp["boxes"][0].set_facecolor(PALETTE[0])
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(PALETTE[1])
        bp["boxes"][1].set_alpha(0.7)

        ax.set_title(col)
        ax.set_ylabel("Valor")

    plt.tight_layout()
    _save_figure("03_boxplots_by_class.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Heatmap da matriz de correlação de Pearson.

    Por que importa:
        - Detecta multicolinearidade (features altamente correlacionadas
          carregam informação redundante)
        - Identifica quais features mais se correlacionam com Outcome
        - Correlação de Pearson mede relação LINEAR entre variáveis

    Interpretação:
        - r próximo de +1: correlação positiva forte
        - r próximo de -1: correlação negativa forte
        - r próximo de 0: sem correlação linear
        - |r| > 0.7: multicolinearidade preocupante
    """
    corr_matrix = df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Ocultar triângulo superior

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 9},
    )

    ax.set_title(
        "Matriz de Correlação de Pearson\n(Correlações com Outcome são as mais relevantes)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save_figure("04_correlation_heatmap.png")


def plot_missing_data_pattern(df: pd.DataFrame) -> None:
    """
    Visualiza o padrão de dados ausentes implícitos (zeros problemáticos).

    Por que importa:
        - Dados ausentes raramente são aleatórios em contexto clínico
        - Entender se a ausência se correlaciona com o desfecho é crítico
        - MCAR (Missing Completely at Random) vs MAR vs MNAR afetam
          a estratégia de imputação

    Hipótese clínica: Insulina não medida pode indicar que o médico
    não solicitou o exame (padrão clínico diferente) — potencialmente
    relacionado ao desfecho.
    """
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # Criar DataFrame de ausência (True = dado ausente)
    df_missing = df[zero_as_missing].copy()
    df_missing = (df_missing == 0).astype(int)
    df_missing["Outcome"] = df["Outcome"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Padrão de Dados Ausentes (Zeros Implícitos)", fontsize=13, fontweight="bold")

    # Taxa geral de ausência por feature
    missing_rates = df_missing[zero_as_missing].mean() * 100
    axes[0].barh(
        missing_rates.index,
        missing_rates.values,
        color=["#E53935" if r > 30 else "#FF9800" if r > 5 else "#4CAF50"
               for r in missing_rates.values],
    )
    axes[0].set_title("Taxa de Ausência por Feature")
    axes[0].set_xlabel("% de Zeros Problemáticos")
    for i, v in enumerate(missing_rates.values):
        axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    axes[0].axvline(30, color="red", linestyle="--", alpha=0.5, label="Limiar crítico (30%)")
    axes[0].legend()

    # Taxa de ausência por classe
    missing_by_class = df_missing.groupby("Outcome")[zero_as_missing].mean() * 100
    missing_by_class.T.plot(
        kind="bar",
        ax=axes[1],
        color=[PALETTE[0], PALETTE[1]],
        alpha=0.8,
        edgecolor="white",
    )
    axes[1].set_title("Taxa de Ausência por Feature e Classe")
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("% de Dados Ausentes")
    axes[1].legend(["Negativo (0)", "Positivo (1)"])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30)

    plt.tight_layout()
    _save_figure("05_missing_data_pattern.png")


def plot_pairplot_key_features(df: pd.DataFrame) -> None:
    """
    Pairplot das features com maior poder discriminativo.

    Features selecionadas com base na correlação com Outcome
    e relevância clínica (Glucose, BMI, Age, DiabetesPedigreeFunction).

    Por que importa:
        - Visualiza relações bivariadas simultaneamente
        - Identifica clusters e padrões entre grupos
        - Revela separabilidade linear vs não-linear
    """
    key_features = ["Glucose", "BMI", "Age", "DiabetesPedigreeFunction", "Outcome"]
    df_subset = df[key_features].copy()

    # Remover zeros problemáticos para visualização
    df_subset = df_subset[(df_subset["Glucose"] > 0) & (df_subset["BMI"] > 0)]

    # Adicionar label textual para a legenda
    df_subset["Diagnóstico"] = df_subset["Outcome"].map({
        0: "Negativo", 1: "Positivo"
    })

    g = sns.pairplot(
        df_subset.drop(columns=["Outcome"]),
        hue="Diagnóstico",
        palette={"Negativo": PALETTE[0], "Positivo": PALETTE[1]},
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 20},
        height=2.5,
    )

    g.figure.suptitle(
        "Pairplot — Features de Maior Relevância Clínica",
        fontsize=13, fontweight="bold", y=1.02,
    )

    _save_figure("06_pairplot_key_features.png")


def compute_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela comparativa de estatísticas descritivas por classe.

    Permite identificar quais features diferem mais entre grupos,
    orientando a seleção de features para modelagem.
    """
    groups = df.groupby("Outcome")[FEATURE_COLS]
    summary = groups.agg(["mean", "median", "std"]).round(2)

    print("\n" + "=" * 70)
    print("ESTATÍSTICAS DESCRITIVAS POR CLASSE")
    print("=" * 70)
    print(summary.to_string())

    # Diferença percentual entre médias (efeito clínico)
    print("\n" + "-" * 40)
    print("DIFERENÇA PERCENTUAL DAS MÉDIAS (Positivo vs Negativo):")
    mean_neg = df[df["Outcome"] == 0][FEATURE_COLS].mean()
    mean_pos = df[df["Outcome"] == 1][FEATURE_COLS].mean()
    diff_pct = ((mean_pos - mean_neg) / mean_neg * 100).round(1)
    diff_df = pd.DataFrame({
        "Média Negativo": mean_neg.round(2),
        "Média Positivo": mean_pos.round(2),
        "Diferença %": diff_pct,
    })
    print(diff_df.to_string())
    print("\nINTERPRETAÇÃO: Diferenças maiores indicam maior poder discriminativo.")

    return summary


def run_eda(df: pd.DataFrame) -> None:
    """
    Executa pipeline completo de EDA.

    Ordem de execução:
        1. Distribuição da variável alvo
        2. Distribuições univariadas por classe
        3. Boxplots comparativos
        4. Matriz de correlação
        5. Padrão de dados ausentes
        6. Pairplot de features-chave
        7. Sumário estatístico
    """
    print("\n" + "=" * 60)
    print("INICIANDO ANÁLISE EXPLORATÓRIA DE DADOS")
    print("=" * 60)

    print("\n[1/7] Distribuição da variável alvo...")
    plot_class_distribution(df)

    print("[2/7] Distribuições por feature...")
    plot_feature_distributions(df)

    print("[3/7] Boxplots por classe...")
    plot_boxplots_by_class(df)

    print("[4/7] Matriz de correlação...")
    plot_correlation_heatmap(df)

    print("[5/7] Padrão de dados ausentes...")
    plot_missing_data_pattern(df)

    print("[6/7] Pairplot features-chave...")
    plot_pairplot_key_features(df)

    print("[7/7] Sumário estatístico...")
    compute_statistical_summary(df)

    print(f"\n[OK] EDA concluída. Figuras salvas em: {FIGURES_DIR}/")


# Execução direta

if __name__ == "__main__":
    df = load_and_validate()
    run_eda(df)