# 📋 Plano de Projeto — Diabetes Risk Prediction

## Metodologia: CRISP-DM Adaptado
Business Understanding
│
▼
Data Understanding
│
▼
Data Preparation
│
▼
Modeling
│
▼
Evaluation
│
▼
Deployment

---

## Fase 1 — Business & Data Understanding

**Objetivo:** Compreender o problema clínico e os dados disponíveis.

- [x] Definir problema de negócio (triagem de DM2)
- [x] Identificar stakeholders (profissionais de saúde, pacientes)
- [x] Criar dicionário de dados com contexto clínico
- [ ] Executar relatório de qualidade dos dados (`data_quality.py`)
- [ ] EDA completa (`src/eda.py`)

**Critério de conclusão:** Documento de qualidade gerado; todos os problemas de dados mapeados.

---

## Fase 2 — Data Preparation

**Objetivo:** Transformar dados brutos em features prontas para modelagem.

- [ ] Substituir zeros impossíveis por NaN
- [ ] Imputar valores ausentes (estratégia documentada)
- [ ] Detectar e tratar outliers
- [ ] Normalizar/escalar features
- [ ] Engenharia de features (faixas etárias, categorias de IMC)
- [ ] Divisão treino/teste estratificada (80/20)
- [ ] Salvar dataset processado em `data/processed/`

**Critério de conclusão:** Dataset limpo sem valores ausentes; escalonamento aplicado; seed fixo para reprodutibilidade.

---

## Fase 3 — Modeling

**Objetivo:** Treinar e comparar múltiplos modelos.

Modelos:
1. Logistic Regression (baseline)
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. K-Nearest Neighbors

Estratégia:
- Validação cruzada estratificada (5-fold)
- GridSearchCV para hiperparâmetros
- Seed fixo: `RANDOM_STATE = 42`

---

## Fase 4 — Evaluation

**Objetivo:** Avaliar modelos com métricas clinicamente relevantes.

Métricas:
- ROC-AUC (discriminação geral)
- Recall/Sensibilidade (falsos negativos = risco clínico)
- F1-Score (balanço em dados desbalanceados)
- Precision (alarmes falsos)
- Matriz de Confusão
- Curva ROC comparativa

---

## Decisões Técnicas e Justificativas

| Decisão | Justificativa |
|---------|--------------|
| `RANDOM_STATE = 42` | Reprodutibilidade universal |
| Divisão 80/20 estratificada | Mantém proporção de classes em ambos os conjuntos |
| Imputação por mediana | Robusta a outliers; preferível à média em dados clínicos assimétricos |
| Escalonamento StandardScaler | Necessário para KNN e Logistic Regression; inofensivo para tree-based |
| Métrica primária: ROC-AUC | Independente de threshold; ideal para comparação de modelos |
| Métrica de decisão: Recall | Em triagem, falso negativo tem consequência clínica grave |

---

## Riscos e Limitações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| Alta taxa de ausência em Insulin (48%) | Alta | Médio | Feature flag + análise de sensibilidade |
| Dataset pequeno (768 obs) | Alta | Alto | Validação cruzada; evitar overfitting |
| Viés de população (apenas mulheres Pima) | Alta | Alto | Documentar limitação explicitamente |
| Desbalanceamento de classes | Média | Médio | Usar F1/AUC; avaliar class_weight='balanced' |

---

## Cronograma

| Semana | Atividade |
|--------|-----------|
| 1 | Estrutura, documentação, qualidade dos dados |
| 2 | EDA completa + pré-processamento |
| 3 | Treinamento e avaliação dos modelos |
| 4 | Interpretação, relatório final, notebook de portfólio |