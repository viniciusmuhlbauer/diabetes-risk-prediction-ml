# 📖 Dicionário de Dados — Pima Indians Diabetes Database

## Metadados do Dataset

| Campo | Valor |
|-------|-------|
| Fonte | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) |
| Repositório | UCI ML Repository / Kaggle |
| Arquivo | `data/raw/pima_indians_diabetes.csv` |
| Observações | 768 |
| Features | 8 preditoras + 1 alvo |
| Tipo de problema | Classificação binária supervisionada |
| População | Mulheres, ≥ 21 anos, herança indígena Pima (Arizona, EUA) |

---

## Variáveis Preditoras

### 1. `Pregnancies` — Número de Gestações
- **Tipo:** Inteiro (discreta)
- **Unidade:** Número de vezes
- **Range esperado:** 0 – 17
- **Contexto clínico:** Diabetes gestacional é um fator de risco conhecido para DM2. Cada gestação aumenta transitoriamente a resistência à insulina. Mulheres com histórico de diabetes gestacional têm risco 7x maior de desenvolver DM2.
- **Valores ausentes codificados como zero:** Biologicamente possível (nunca engravidou). Zeros são válidos.
- **Ação:** Nenhuma imputação necessária para zeros.

---

### 2. `Glucose` — Concentração de Glicose Plasmática em Jejum
- **Tipo:** Contínua
- **Unidade:** mg/dL (miligramas por decilitro)
- **Range de referência clínica:**
  - Normal: < 100 mg/dL
  - Pré-diabetes: 100–125 mg/dL
  - Diabetes: ≥ 126 mg/dL (ADA, 2023)
- **Range no dataset:** 0 – 199
- **Contexto clínico:** Principal critério diagnóstico de DM2 segundo a American Diabetes Association. É o preditor individual mais forte neste dataset.
- **⚠️ Problema de qualidade:** Valor 0 é biologicamente impossível (glicemia zero = morte). Zeros representam dados ausentes.
- **Ação:** Substituir zeros por NaN → imputar com mediana estratificada por desfecho.

---

### 3. `BloodPressure` — Pressão Arterial Diastólica
- **Tipo:** Contínua
- **Unidade:** mmHg (milímetros de mercúrio)
- **Range de referência clínica:**
  - Normal: < 80 mmHg
  - Elevada: 80–89 mmHg
  - Hipertensão: ≥ 90 mmHg
- **Range no dataset:** 0 – 122
- **Contexto clínico:** Hipertensão e DM2 coexistem com frequência (síndrome metabólica). Pressão diastólica elevada cronicamente indica resistência vascular periférica.
- **⚠️ Problema de qualidade:** Valor 0 é biologicamente impossível em pessoa viva.
- **Ação:** Substituir zeros por NaN → imputar com mediana.

---

### 4. `SkinThickness` — Espessura da Dobra Cutânea do Tríceps
- **Tipo:** Contínua
- **Unidade:** mm (milímetros)
- **Range esperado:** 10 – 50 mm (adultas)
- **Contexto clínico:** Proxy de gordura subcutânea e resistência à insulina. Utilizado em estimativas de composição corporal quando não há acesso à bioimpedância.
- **⚠️ Problema de qualidade:** 29,6% dos valores são 0 — impossível biologicamente.
- **Ação:** Alta taxa de ausência. Imputar com mediana ou KNN; avaliar impacto no modelo com e sem a feature.

---

### 5. `Insulin` — Nível de Insulina Sérica (2h)
- **Tipo:** Contínua
- **Unidade:** μU/mL (microunidades por mililitro)
- **Range de referência clínica:** 16 – 166 μU/mL (2h pós-sobrecarga de glicose)
- **Contexto clínico:** Insulina elevada indica hiperinsulinemia compensatória — sinal clássico de resistência à insulina que precede o DM2. Insulina muito baixa pode indicar disfunção pancreática.
- **⚠️ Problema de qualidade:** 48,7% dos valores são 0 — a variável com maior taxa de ausência.
- **Ação:** Tratar com cautela. Alta imputation uncertainty. Considerar exclusão ou flag binário de "dado disponível".

---

### 6. `BMI` — Índice de Massa Corporal
- **Tipo:** Contínua
- **Unidade:** kg/m²
- **Range de referência clínica (OMS):**
  - Abaixo do peso: < 18,5
  - Normal: 18,5 – 24,9
  - Sobrepeso: 25,0 – 29,9
  - Obesidade grau I: 30,0 – 34,9
  - Obesidade grau II: 35,0 – 39,9
  - Obesidade grau III: ≥ 40,0
- **Contexto clínico:** IMC ≥ 30 é o principal fator de risco modificável para DM2. Obesidade central amplifica a resistência à insulina.
- **⚠️ Problema de qualidade:** 11 valores são 0 — impossível.
- **Ação:** Substituir zeros por NaN → imputar com mediana.

---

### 7. `DiabetesPedigreeFunction` — Função de Pedigree Diabético
- **Tipo:** Contínua
- **Unidade:** Adimensional (score calculado)
- **Range no dataset:** 0.078 – 2.42
- **Contexto clínico:** Score proprietário que quantifica a história familiar de diabetes, ponderando grau de parentesco e expressão da doença em familiares. Valores mais altos indicam maior carga genética.
- **Valores ausentes codificados como zero:** Não aplicável — zero seria score mínimo válido no sistema, mas biologicamente improvável. Verificar durante EDA.
- **Ação:** Monitorar distribuição; verificar outliers superiores.

---

### 8. `Age` — Idade
- **Tipo:** Inteiro (discreta)
- **Unidade:** Anos
- **Range no dataset:** 21 – 81
- **Contexto clínico:** Risco de DM2 aumenta progressivamente com a idade devido ao declínio da função das células beta pancreáticas e aumento da resistência à insulina. Após os 45 anos, o rastreamento é recomendado pela ADA.
- **Valores ausentes:** Nenhum esperado.
- **Ação:** Considerar criação de faixas etárias (feature engineering).

---

## Variável Alvo

### 9. `Outcome` — Diagnóstico de Diabetes
- **Tipo:** Binária (0 ou 1)
- **Encoding:**
  - `0` = Negativo para diabetes (dentro de 5 anos após a coleta)
  - `1` = Positivo para diabetes (dentro de 5 anos após a coleta)
- **Distribuição:**
  - Classe 0: 500 observações (65,1%)
  - Classe 1: 268 observações (34,9%)
- **Contexto clínico:** Diagnóstico realizado pelo NIDDK dentro de um período de acompanhamento de 5 anos. Não especifica critério diagnóstico utilizado (glicemia de jejum, TOTG ou HbA1c).
- **Observação sobre desbalanceamento:** Proporção 65/35 é moderada. Não exige técnicas de oversampling agressivas, mas deve ser considerada na escolha da métrica de avaliação (preferir F1 e AUC-ROC sobre accuracy).

---

## Mapa de Qualidade

| Feature | Zeros Problemáticos | % Ausência Real | Ação Recomendada |
|---------|--------------------|-----------------|--------------------|
| Pregnancies | Válidos | 0% | Nenhuma |
| Glucose | ⚠️ Sim (5) | 0,7% | Imputar mediana |
| BloodPressure | ⚠️ Sim (35) | 4,6% | Imputar mediana |
| SkinThickness | ⚠️ Sim (227) | 29,6% | Imputar mediana/KNN |
| Insulin | ⚠️ Sim (374) | 48,7% | Flag + imputar mediana |
| BMI | ⚠️ Sim (11) | 1,4% | Imputar mediana |
| DiabetesPedigreeFunction | Verificar | ~0% | Monitorar |
| Age | Não | 0% | Nenhuma |
| Outcome | — | 0% | Variável alvo |

---

## Referências Clínicas

- American Diabetes Association. *Standards of Medical Care in Diabetes*, 2023.
- World Health Organization. *Diabetes Fact Sheet*, 2023.
- Smith, J.W. et al. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.