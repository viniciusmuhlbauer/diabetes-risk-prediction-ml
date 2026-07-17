"""
Avaliação completa e interpretação dos modelos treinados.

Responsabilidade:
    - Calcular métricas de classificação no conjunto de teste
    - Gerar matrizes de confusão
    - Plotar curvas ROC comparativas
    - Analisar importância de features
    - Gerar relatório final de avaliação

Métricas e Justificativas Clínicas:

    AUC-ROC:
        Mede a capacidade discriminativa do modelo independente do threshold.
        AUC = 0.5 → modelo aleatório | AUC = 1.0 → modelo perfeito
        Ideal para comparação entre modelos.

    Recall (Sensibilidade):
        TP / (TP + FN)
        Em triagem: PRIORIDADE MÁXIMA.
        Falso Negativo = dizer que um diabético não tem diabetes.
        Consequência clínica: paciente não recebe tratamento preventivo.

    Precisão (Positive Predictive Value):
        TP / (TP + FP)
        Falso Positivo = alarme desnecessário.
        Consequência: exames adicionais, custo, ansiedade do paciente.

    F1-Score:
        Média harmônica de Precisão e Recall.
        Métrica balanceada para classes desbalanceadas.

    Especificidade:
        TN / (TN + FP) — capacidade de identificar corretamente negativos.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)


# Configurações

MODELS_DIR = "models"
FIGURES_DIR = os.path.join("outputs", "figures")
REPORTS_DIR = os.path.join("outputs", "reports")
RANDOM_STATE = 42

PALETTE = {
    "LogisticRegression": "#2196F3",
    "DecisionTree": "#4CAF50",
    "RandomForest": "#FF9800",
    "GradientBoosting": "#E53935",
    "KNN": "#9C27B0",
}

MODEL_NAMES = list(PALETTE.keys())


# Carregamento


def load_test_data():
    """Carrega conjunto de teste processado."""
    X_test = pd.read_csv(os.path.join("data", "processed", "X_test.csv"), index_col=0)
    y_test = pd.read_csv(os.path.join("data", "processed", "y_test.csv"), index_col=0).squeeze()
    return X_test, y_test


def load_trained_models() -> dict:
    """Carrega todos os modelos serializados."""
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{name.lower()}_best.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"  [OK] Modelo carregado: {name}")
        else:
            print(f"  [AVISO] Modelo não encontrado: {path}")
    return models


# Métricas


def compute_metrics(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Calcula métricas completas para todos os modelos.

    Inclui cálculo de especificidade (TN rate) — métrica clínica importante
    não incluída diretamente no sklearn.classification_report.

    Parâmetros
    ----------
    models : dict
    X_test : pd.DataFrame
    y_test : pd.Series

    Retorna
    -------
    pd.DataFrame
        Tabela de métricas por modelo, ordenada por AUC-ROC.
    """
    rows = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidade da classe positiva

        # Matriz de confusão
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Especificidade = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC-ROC": roc_auc_score(y_test, y_proba),
            "Recall (Sens.)": recall_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "Specificity": specificity,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        })

    df_metrics = pd.DataFrame(rows)
    df_metrics = df_metrics.sort_values("AUC-ROC", ascending=False)

    # Arredondar métricas para exibição
    metric_cols = ["Accuracy", "AUC-ROC", "Recall (Sens.)", "Precision", "F1-Score", "Specificity"]
    df_metrics[metric_cols] = df_metrics[metric_cols].round(4)

    return df_metrics


# Visualizações


def plot_confusion_matrices(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Matrizes de confusão para todos os modelos.

    Interpretação clínica:
        TP (Verdadeiro Positivo): diabético identificado corretamente
        TN (Verdadeiro Negativo): não-diabético identificado corretamente
        FP (Falso Positivo): alarme falso — exames desnecessários
        FN (Falso Negativo): diabético não detectado — RISCO CLÍNICO MAIOR

    Por que normalizar por linha:
        Permite ver a taxa de cada tipo de erro independente do
        tamanho de cada classe.
    """
    n_models = len(models)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Matrizes de Confusão — Conjunto de Teste", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    labels = ["Negativo (0)", "Positivo (1)"]

    for i, (name, model) in enumerate(models.items()):
        ax = axes[i]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]  # Normalização por linha

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

        # Anotações com contagem absoluta e percentual
        for row in range(2):
            for col in range(2):
                count = cm[row, col]
                pct = cm_norm[row, col]
                color = "white" if pct > 0.5 else "black"
                ax.text(col, row, f"{count}\n({pct:.1%})",
                        ha="center", va="center", color=color, fontsize=11, fontweight="bold")

        ax.set_title(name, fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")

    # Remover subplot extra se houver
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.colorbar(im, ax=axes[:n_models], label="Taxa (por classe real)")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "07_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[OK] Figura salva: {path}")
    plt.close()


def plot_roc_curves(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Curvas ROC comparativas para todos os modelos.

    A curva ROC plota:
        - Eixo X: Taxa de Falso Positivo (1 - Especificidade)
        - Eixo Y: Taxa de Verdadeiro Positivo (Recall/Sensibilidade)

    Para cada threshold de decisão, um ponto na curva.
    A linha diagonal (AUC=0.5) representa um classificador aleatório.
    Quanto maior a área sob a curva, melhor o modelo.

    Por que AUC-ROC é preferível à accuracy:
        - Independente do threshold de decisão
        - Robusto a desbalanceamento de classes
        - Mostra o trade-off sensibilidade × especificidade
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})",
                color=PALETTE.get(name, "gray"), linewidth=2)

    # Linha de referência (classificador aleatório)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Aleatório (AUC=0.50)", alpha=0.6)

    ax.set_xlabel("Taxa de Falso Positivo (1 - Especificidade)", fontsize=11)
    ax.set_ylabel("Taxa de Verdadeiro Positivo (Sensibilidade)", fontsize=11)
    ax.set_title("Curvas ROC — Comparação de Modelos\n(Conjunto de Teste)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1.02))

    # Área de "bom" desempenho (AUC > 0.8)
    ax.fill_between([0, 1], [0.8, 0.8], [1, 1], alpha=0.05, color="green")
    ax.axhline(0.8, color="green", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(0.02, 0.81, "AUC > 0.8 (bom)", color="green", fontsize=8, alpha=0.7)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "08_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[OK] Figura salva: {path}")
    plt.close()


def plot_feature_importance(models: dict, feature_names: list) -> None:
    """
    Importância de features para modelos baseados em árvore.

    Random Forest e Gradient Boosting fornecem importância de features
    como o ganho médio de impureza (Gini) ao dividir em cada feature.

    Feature importance vs. coeficientes de Logistic Regression:
        - Tree-based: mede ganho de informação não-linear
        - LR: mede contribuição linear para o log-odds

    Interpretação clínica:
        Features com alta importância são as principais "sinalizadoras"
        clínicas segundo o modelo — devem ser correlacionadas com
        o conhecimento médico estabelecido para validação.
    """
    tree_models = {
        name: model for name, model in models.items()
        if hasattr(model, "feature_importances_")
    }

    if not tree_models:
        print("  Nenhum modelo baseado em árvore disponível para importância de features.")
        return

    n = len(tree_models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    fig.suptitle("Importância de Features (Modelos Baseados em Árvore)",
                 fontsize=13, fontweight="bold")

    for ax, (name, model) in zip(axes, tree_models.items()):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        colors = [PALETTE.get(name, "#607D8B")] * len(sorted_names)
        colors[0] = "#E53935"  # Destaque para a feature mais importante

        ax.barh(sorted_names[::-1], sorted_importances[::-1],
                color=colors[::-1], edgecolor="white")
        ax.set_title(name)
        ax.set_xlabel("Importância (Ganho de Impureza Médio)")
        ax.set_xlim(0, max(sorted_importances) * 1.15)

        for i, (imp, feat) in enumerate(zip(sorted_importances[::-1], sorted_names[::-1])):
            ax.text(imp + 0.002, i, f"{imp:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "09_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[OK] Figura salva: {path}")
    plt.close()


def plot_metrics_comparison(metrics_df: pd.DataFrame) -> None:
    """
    Gráfico comparativo das principais métricas por modelo.
    """
    metric_cols = ["AUC-ROC", "Recall (Sens.)", "Precision", "F1-Score", "Specificity"]
    df_plot = metrics_df.set_index("Model")[metric_cols]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df_plot))
    width = 0.15
    colors = ["#1565C0", "#C62828", "#2E7D32", "#F57F17", "#6A1B9A"]

    for i, (col, color) in enumerate(zip(metric_cols, colors)):
        offset = (i - len(metric_cols) / 2) * width
        bars = ax.bar(x + offset, df_plot[col], width, label=col, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Comparação de Métricas por Modelo\n(Conjunto de Teste)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(len(x) - 0.5, 0.81, "0.80", color="gray", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "10_metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[OK] Figura salva: {path}")
    plt.close()


# Relatório Final


def generate_evaluation_report(metrics_df: pd.DataFrame) -> None:
    """Gera relatório textual completo de avaliação."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "evaluation_report.txt")

    best_auc = metrics_df.iloc[0]
    best_recall = metrics_df.loc[metrics_df["Recall (Sens.)"].idxmax()]

    lines = [
        "=" * 70,
        "RELATÓRIO DE AVALIAÇÃO DOS MODELOS",
        "Projeto: Diabetes Risk Prediction",
        "=" * 70,
        "",
        "MÉTRICAS COMPLETAS (Conjunto de Teste):",
        "-" * 40,
        metrics_df[["Model", "AUC-ROC", "Recall (Sens.)", "Precision",
                     "F1-Score", "Specificity", "TP", "FP", "TN", "FN"]].to_string(index=False),
        "",
        "MODELO RECOMENDADO (maior AUC-ROC):",
        f"  {best_auc['Model']}",
        f"  AUC-ROC:  {best_auc['AUC-ROC']:.4f}",
        f"  Recall:   {best_auc['Recall (Sens.)']:.4f}",
        f"  F1-Score: {best_auc['F1-Score']:.4f}",
        "",
        "MODELO DE MAIOR SENSIBILIDADE (triagem conservadora):",
        f"  {best_recall['Model']}",
        f"  Recall:   {best_recall['Recall (Sens.)']:.4f}",
        f"  FN (diabéticos não detectados): {int(best_recall['FN'])}",
        "",
        "ANÁLISE DE ERROS CLÍNICOS:",
        "-" * 40,
    ]

    for _, row in metrics_df.iterrows():
        total_positives = int(row["TP"] + row["FN"])
        fn_rate = int(row["FN"]) / total_positives if total_positives > 0 else 0
        lines.append(
            f"  {row['Model']:<25} "
            f"FN={int(row['FN'])} ({fn_rate:.1%} dos diabéticos não detectados) | "
            f"FP={int(row['FP'])}"
        )

    lines += [
        "",
        "LIMITAÇÕES DO PROJETO:",
        "-" * 40,
        "1. Dataset restrito: 768 observações — pequeno para generalização clínica",
        "2. Viés de população: apenas mulheres ≥21 anos com herança indígena Pima",
        "3. Alta taxa de ausência em Insulin (48.7%) — imputação introduz incerteza",
        "4. Desfecho: diagnóstico em 5 anos — não aplicável para diagnóstico atual",
        "5. Sem validação externa em populações independentes",
        "6. Modelos não calibrados — probabilidades podem ser imprecisas",
        "",
        "RECOMENDAÇÕES PARA USO CLÍNICO:",
        "-" * 40,
        "• NÃO utilizar como ferramenta diagnóstica independente",
        "• Uso como ferramenta de TRIAGEM auxiliar ao julgamento clínico",
        "• Priorizar Recall sobre Precision em contexto de triagem",
        "• Validar em população local antes de implantação",
        "• Monitorar performance e recalibrar periodicamente",
        "",
        "=" * 70,
        "FIM DO RELATÓRIO",
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[OK] Relatório salvo em: {report_path}")


# Pipeline Principal


def run_evaluation():
    """Pipeline completo de avaliação."""
    print("\n" + "=" * 60)
    print("INICIANDO AVALIAÇÃO DOS MODELOS")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("\n[1/6] Carregando dados e modelos...")
    X_test, y_test = load_test_data()
    models = load_trained_models()

    if not models:
        print("[ERRO] Nenhum modelo encontrado. Execute src/train.py primeiro.")
        return

    print(f"\n  Conjunto de teste: {X_test.shape[0]} obs × {X_test.shape[1]} features")
    print(f"  Modelos carregados: {list(models.keys())}")

    print("\n[2/6] Calculando métricas...")
    metrics_df = compute_metrics(models, X_test, y_test)

    print("\n[3/6] Matrizes de confusão...")
    plot_confusion_matrices(models, X_test, y_test)

    print("[4/6] Curvas ROC...")
    plot_roc_curves(models, X_test, y_test)

    print("[5/6] Importância de features...")
    plot_feature_importance(models, X_test.columns.tolist())

    print("[5b/6] Comparação de métricas...")
    plot_metrics_comparison(metrics_df)

    print("\n[6/6] Relatório de avaliação...")
    generate_evaluation_report(metrics_df)

    print("\n[OK] Avaliação concluída.")
    return metrics_df, models


# Execução direta

if __name__ == "__main__":
    run_evaluation()