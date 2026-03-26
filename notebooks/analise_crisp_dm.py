#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
  CLASSIFICADOR DE RISCO CARDIOVASCULAR — PIPELINE CRISP-DM COMPLETO
=============================================================================
  Projeto: Desafio Final — UC Aprendizado de Máquina 2026/1
  Dataset: Saúde Brasil (20.000 registros simulados)
  Referência: Sociedade Brasileira de Cardiologia — Escore de Risco Global

  Este conteúdo é destinado apenas para fins educacionais.
  Os dados exibidos são ilustrativos e podem não corresponder a situações reais.
=============================================================================
"""

import os
import sys
import warnings
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import stats
from scipy.stats import shapiro, kruskal, chi2_contingency, f_oneway, spearmanr

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')

# =============================================================================
#  CONFIGURAÇÕES
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset_saude_brasil.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'notebooks', 'figures')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Estilo dos gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

RANDOM_STATE = 42
RISK_ORDER = ['Baixo', 'Moderado', 'Alto', 'Muito Alto']
RISK_COLORS = {'Baixo': '#27ae60', 'Moderado': '#f39c12', 'Alto': '#e74c3c', 'Muito Alto': '#8e44ad'}

print("=" * 80)
print(" CLASSIFICADOR DE RISCO CARDIOVASCULAR — PIPELINE CRISP-DM")
print("=" * 80)

# =============================================================================
#  FASE 1 — BUSINESS UNDERSTANDING (Entendimento do Negócio)
# =============================================================================
print("\n" + "=" * 80)
print(" FASE 1 — BUSINESS UNDERSTANDING")
print("=" * 80)

print("""
CONTEXTO:
  A Sociedade Brasileira de Cardiologia (SBC) define a estratificação do risco
  cardiovascular como processo fundamental para avaliar a probabilidade de
  eventos cardíacos adversos em 10 anos, utilizando o Escore de Risco Global
  (ERG) de Framingham.

OBJETIVO:
  Construir um classificador de machine learning que, com base em dados
  clínicos e estilo de vida, classifique pacientes em 4 níveis de risco:
    • Baixo: probabilidade < 5%
    • Moderado: probabilidade 5-10% (mulheres) / 5-20% (homens)
    • Alto: probabilidade > 10% (mulheres) / > 20% (homens)
    • Muito Alto: eventos cardiovasculares prévios / condições graves

FATORES CONSIDERADOS (SBC):
  - Idade, Sexo, IMC, Pressão Arterial, Colesterol
  - Frequência Cardíaca em Repouso
  - Tabagismo, Consumo de Álcool
  - Nível de atividade física (passos diários)
  - Histórico Familiar de DCV
""")

# =============================================================================
#  FASE 2 — DATA UNDERSTANDING (Entendimento dos Dados)
# =============================================================================
print("\n" + "=" * 80)
print(" FASE 2 — DATA UNDERSTANDING")
print("=" * 80)

# 2.1 Carregamento
print("\n--- 2.1 Carregamento dos Dados ---")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Registros: {df.shape[0]:,}")
print(f"  Features: {df.shape[1]}")

# 2.2 Primeiras observações
print("\n--- 2.2 Primeiras Observações ---")
print(df.head(10).to_string())
print("\nTipos de dados:")
print(df.dtypes.to_string())

# 2.3 Conversão de tipos problemáticos
print("\n--- 2.3 Conversão de Tipos ---")
cols_to_numeric = ['Passos_Diarios', 'Calorias', 'Colesterol']
for col in cols_to_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"  {col}: convertido para numérico")

# Definir colunas numéricas e categóricas
num_cols = df.select_dtypes(include=[np.number]).columns.drop('ID').tolist()
cat_cols = ['Sexo', 'Fumante', 'Alcool', 'Historico_Familiar']
target = 'Risco_Doenca'

print(f"\n  Colunas numéricas ({len(num_cols)}): {num_cols}")
print(f"  Colunas categóricas ({len(cat_cols)}): {cat_cols}")

# 2.4 Análise de valores faltantes
print("\n--- 2.4 Análise de Valores Faltantes ---")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Contagem': missing, 'Percentual (%)': missing_pct})
missing_df = missing_df[missing_df['Contagem'] > 0]
print(missing_df.to_string())

# Gráfico de missing values
fig, ax = plt.subplots(figsize=(10, 5))
if not missing_df.empty:
    colors = ['#e74c3c' if p > 1 else '#f39c12' for p in missing_df['Percentual (%)']]
    missing_df['Percentual (%)'].plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.set_xlabel('Percentual de valores faltantes (%)')
    ax.set_title('Valores Faltantes por Variável', fontweight='bold')
    for i, (val, name) in enumerate(zip(missing_df['Percentual (%)'], missing_df.index)):
        ax.text(val + 0.05, i, f'{val:.1f}%', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '01_missing_values.png'))
plt.close()
print("  ✓ Gráfico salvo: 01_missing_values.png")

# 2.5 Estatísticas descritivas avançadas
print("\n--- 2.5 Estatísticas Descritivas Avançadas ---")
desc_stats = df[num_cols].describe().T
desc_stats['skewness'] = df[num_cols].skew()
desc_stats['kurtosis'] = df[num_cols].kurtosis()
desc_stats['cv (%)'] = (desc_stats['std'] / desc_stats['mean'] * 100).round(2)
desc_stats['iqr'] = desc_stats['75%'] - desc_stats['25%']
print(desc_stats.round(2).to_string())

# Salvar estatísticas
desc_stats.round(4).to_csv(os.path.join(FIGURES_DIR, 'estatisticas_descritivas.csv'))
print("  ✓ Estatísticas salvas: estatisticas_descritivas.csv")

# 2.6 Distribuição da variável target
print("\n--- 2.6 Distribuição da Variável Target ---")
target_counts = df[target].value_counts()
target_pct = (df[target].value_counts(normalize=True) * 100).round(2)
print(pd.DataFrame({'Contagem': target_counts, 'Percentual (%)': target_pct}).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = [RISK_COLORS[r] for r in RISK_ORDER]

# Barplot
order_counts = [target_counts.get(r, 0) for r in RISK_ORDER]
axes[0].bar(RISK_ORDER, order_counts, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_title('Distribuição dos Níveis de Risco', fontweight='bold')
axes[0].set_ylabel('Contagem')
for i, (v, r) in enumerate(zip(order_counts, RISK_ORDER)):
    axes[0].text(i, v + 100, f'{v:,}\n({target_pct.get(r, 0):.1f}%)', ha='center', fontsize=10)

# Pie chart
axes[1].pie(order_counts, labels=RISK_ORDER, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=[0, 0, 0.05, 0.1], shadow=True,
            textprops={'fontsize': 11})
axes[1].set_title('Proporção dos Níveis de Risco', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '02_distribuicao_target.png'))
plt.close()
print("  ✓ Gráfico salvo: 02_distribuicao_target.png")

# 2.7 Distribuições univariadas
print("\n--- 2.7 Distribuições Univariadas ---")
n_num = len(num_cols)
n_cols_plot = 3
n_rows_plot = (n_num + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(18, 4 * n_rows_plot))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    for risk in RISK_ORDER:
        subset = df[df[target] == risk][col].dropna()
        if len(subset) > 0:
            ax.hist(subset, bins=30, alpha=0.5, label=risk, color=RISK_COLORS[risk], density=True)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('')
    if i == 0:
        ax.legend(fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribuições Univariadas por Nível de Risco', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '03_distribuicoes_univariadas.png'))
plt.close()
print("  ✓ Gráfico salvo: 03_distribuicoes_univariadas.png")

# 2.8 Boxplots por classe de risco
print("\n--- 2.8 Boxplots por Classe de Risco ---")
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(18, 4 * n_rows_plot))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    data_bp = [df[df[target] == r][col].dropna().values for r in RISK_ORDER]
    bp = ax.boxplot(data_bp, labels=RISK_ORDER, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(col, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Boxplots por Nível de Risco Cardiovascular', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '04_boxplots_por_risco.png'))
plt.close()
print("  ✓ Gráfico salvo: 04_boxplots_por_risco.png")

# 2.9 Matriz de correlação
print("\n--- 2.9 Matriz de Correlação ---")
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Pearson
corr_pearson = df[num_cols].corr(method='pearson')
mask = np.triu(np.ones_like(corr_pearson, dtype=bool))
sns.heatmap(corr_pearson, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=axes[0], square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
axes[0].set_title('Correlação de Pearson', fontweight='bold')

# Spearman
corr_spearman = df[num_cols].corr(method='spearman')
sns.heatmap(corr_spearman, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=axes[1], square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
axes[1].set_title('Correlação de Spearman', fontweight='bold')

plt.suptitle('Matrizes de Correlação', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '05_correlacao.png'))
plt.close()
print("  ✓ Gráfico salvo: 05_correlacao.png")

# 2.10 Testes estatísticos
print("\n--- 2.10 Testes Estatísticos ---")

# Teste de normalidade (D'Agostino-Pearson para N > 5000)
print("\n  [A] Teste de Normalidade (D'Agostino-Pearson):")
normality_results = []
for col in num_cols:
    data = df[col].dropna()
    if len(data) > 8:
        stat, p_value = stats.normaltest(data)
        normal = 'Sim' if p_value > 0.05 else 'Não'
        normality_results.append({'Variável': col, 'Estatística': round(stat, 4),
                                  'p-valor': f'{p_value:.2e}', 'Normal (α=0.05)': normal})
        print(f"    {col}: stat={stat:.4f}, p={p_value:.2e} → {'Normal' if p_value > 0.05 else 'Não Normal'}")

normality_df = pd.DataFrame(normality_results)

# Kruskal-Wallis (variáveis numéricas vs target)
print("\n  [B] Kruskal-Wallis (Numérica vs. Risco):")
kruskal_results = []
for col in num_cols:
    groups = [df[df[target] == r][col].dropna().values for r in RISK_ORDER]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        stat, p_value = kruskal(*groups)
        sig = 'Sim' if p_value < 0.05 else 'Não'
        kruskal_results.append({'Variável': col, 'Estatística H': round(stat, 4),
                                'p-valor': f'{p_value:.2e}', 'Significativo (α=0.05)': sig})
        print(f"    {col}: H={stat:.4f}, p={p_value:.2e} → {'Significativo' if p_value < 0.05 else 'Não Significativo'}")

kruskal_df = pd.DataFrame(kruskal_results)

# Qui-Quadrado (variáveis categóricas vs target)
print("\n  [C] Qui-Quadrado (Categórica vs. Risco):")
chi2_results = []
for col in cat_cols:
    contingency = pd.crosstab(df[col], df[target])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    sig = 'Sim' if p_value < 0.05 else 'Não'
    cramers_v = np.sqrt(chi2 / (len(df) * (min(contingency.shape) - 1)))
    chi2_results.append({'Variável': col, 'χ²': round(chi2, 4), 'p-valor': f'{p_value:.2e}',
                         'gl': dof, "Cramér's V": round(cramers_v, 4),
                         'Significativo (α=0.05)': sig})
    print(f"    {col}: χ²={chi2:.4f}, p={p_value:.2e}, V={cramers_v:.4f} → {'Significativo' if p_value < 0.05 else 'Não Significativo'}")

chi2_df = pd.DataFrame(chi2_results)

# Salvar resultados
test_results = {
    'normalidade': normality_results,
    'kruskal_wallis': kruskal_results,
    'qui_quadrado': chi2_results
}
with open(os.path.join(FIGURES_DIR, 'testes_estatisticos.json'), 'w', encoding='utf-8') as f:
    json.dump(test_results, f, ensure_ascii=False, indent=2)
print("\n  ✓ Resultados salvos: testes_estatisticos.json")

# 2.11 Análise de outliers
print("\n--- 2.11 Análise de Outliers (Método IQR) ---")
outlier_summary = []
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    pct = round(n_outliers / len(df) * 100, 2)
    outlier_summary.append({'Variável': col, 'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
                            'Limite Inf': round(lower, 2), 'Limite Sup': round(upper, 2),
                            'N Outliers': n_outliers, '% Outliers': pct})
    print(f"  {col}: {n_outliers} outliers ({pct}%) | Limites: [{lower:.1f}, {upper:.1f}]")

outlier_df = pd.DataFrame(outlier_summary)

# 2.12 Análise de variáveis categóricas vs risco
print("\n--- 2.12 Variáveis Categóricas vs. Risco ---")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ax = axes[i]
    ct = pd.crosstab(df[col], df[target], normalize='index')[RISK_ORDER] * 100
    ct.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='white')
    ax.set_title(f'{col} vs. Risco Cardiovascular', fontweight='bold')
    ax.set_ylabel('Proporção (%)')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Risco', bbox_to_anchor=(1.0, 1.0), fontsize=8)

plt.suptitle('Variáveis Categóricas vs. Nível de Risco', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '06_categoricas_vs_risco.png'))
plt.close()
print("  ✓ Gráfico salvo: 06_categoricas_vs_risco.png")

# 2.13 Pairplot com amostra
print("\n--- 2.13 Pairplot (amostra de 2000) ---")
key_features = ['Idade', 'IMC', 'Pressao_Sistolica', 'Colesterol', 'Frequencia_Cardiaca_Repouso']
sample = df[key_features + [target]].dropna().sample(2000, random_state=RANDOM_STATE)
g = sns.pairplot(sample, hue=target, hue_order=RISK_ORDER,
                 palette=RISK_COLORS, diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20})
g.figure.suptitle('Pairplot — Features Clínicas Principais', fontsize=16, fontweight='bold', y=1.02)
plt.savefig(os.path.join(FIGURES_DIR, '07_pairplot.png'))
plt.close()
print("  ✓ Gráfico salvo: 07_pairplot.png")


# =============================================================================
#  FASE 3 — DATA PREPARATION (Preparação dos Dados)
# =============================================================================
print("\n" + "=" * 80)
print(" FASE 3 — DATA PREPARATION")
print("=" * 80)

df_prep = df.drop('ID', axis=1).copy()

# 3.1 Tratamento de valores faltantes
print("\n--- 3.1 Tratamento de Valores Faltantes ---")
for col in num_cols:
    if col in df_prep.columns:
        n_missing = df_prep[col].isnull().sum()
        if n_missing > 0:
            median_val = df_prep[col].median()
            df_prep[col].fillna(median_val, inplace=True)
            print(f"  {col}: {n_missing} nulos preenchidos com mediana ({median_val})")

for col in cat_cols:
    n_missing = df_prep[col].isnull().sum()
    if n_missing > 0:
        mode_val = df_prep[col].mode()[0]
        df_prep[col].fillna(mode_val, inplace=True)
        print(f"  {col}: {n_missing} nulos preenchidos com moda ({mode_val})")

print(f"\n  Valores faltantes restantes: {df_prep.isnull().sum().sum()}")

# 3.2 Encoding de variáveis categóricas
print("\n--- 3.2 Encoding de Variáveis ---")

# Label encoding para target (ordem)
le_target = LabelEncoder()
le_target.fit(RISK_ORDER)
df_prep['Risco_Encoded'] = le_target.transform(df_prep[target])
print(f"  Target encoding: {dict(zip(RISK_ORDER, le_target.transform(RISK_ORDER)))}")

# Encoding para variáveis categóricas
encoders = {}

# Sexo: One-hot
df_prep['Sexo_Masculino'] = (df_prep['Sexo'] == 'Masculino').astype(int)
print("  Sexo: One-hot encoding (Sexo_Masculino)")

# Fumante: Binary
df_prep['Fumante_Sim'] = (df_prep['Fumante'] == 'Sim').astype(int)
print("  Fumante: Binary encoding (Fumante_Sim)")

# Álcool: Ordinal
alcool_map = {'Baixo': 0, 'Moderado': 1, 'Alto': 2}
df_prep['Alcool_Encoded'] = df_prep['Alcool'].map(alcool_map)
print(f"  Álcool: Ordinal encoding {alcool_map}")

# Histórico Familiar: Binary
df_prep['Historico_Familiar_Sim'] = (df_prep['Historico_Familiar'] == 'Sim').astype(int)
print("  Histórico Familiar: Binary encoding (Historico_Familiar_Sim)")

# Remover colunas originais categóricas e target textual
cols_to_drop = cat_cols + [target]
df_prep.drop(cols_to_drop, axis=1, inplace=True)

# 3.3 Definir features e target
print("\n--- 3.3 Definição de Features e Target ---")
feature_cols = [c for c in df_prep.columns if c != 'Risco_Encoded']
X = df_prep[feature_cols].values
y = df_prep['Risco_Encoded'].values

print(f"  Features ({len(feature_cols)}): {feature_cols}")
print(f"  Target: Risco_Encoded")
print(f"  Shape X: {X.shape}, Shape y: {y.shape}")

# 3.4 Train/Test Split
print("\n--- 3.4 Split Estratificado (80/20) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {X_train.shape[0]:,} amostras")
print(f"  Test: {X_test.shape[0]:,} amostras")

# Distribuição no split
unique, counts_train = np.unique(y_train, return_counts=True)
unique, counts_test = np.unique(y_test, return_counts=True)
print(f"  Train dist: {dict(zip(le_target.inverse_transform(unique), counts_train))}")
print(f"  Test dist:  {dict(zip(le_target.inverse_transform(unique), counts_test))}")

# 3.5 Feature Scaling
print("\n--- 3.5 Feature Scaling (StandardScaler) ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  ✓ StandardScaler ajustado ao conjunto de treino")

# 3.6 SMOTE para balanceamento
print("\n--- 3.6 Balanceamento com SMOTE ---")
print(f"  Antes SMOTE: {dict(zip(le_target.inverse_transform(unique), counts_train))}")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
unique_b, counts_b = np.unique(y_train_balanced, return_counts=True)
print(f"  Após SMOTE:  {dict(zip(le_target.inverse_transform(unique_b), counts_b))}")
print(f"  Train balanceado: {X_train_balanced.shape[0]:,} amostras")


# =============================================================================
#  FASE 4 — MODELING (Modelagem)
# =============================================================================
print("\n" + "=" * 80)
print(" FASE 4 — MODELING")
print("=" * 80)

# 4.1 LazyPredict para comparação rápida
print("\n--- 4.1 LazyPredict — Comparação de Múltiplos Modelos ---")
try:
    from lazypredict.Supervised import LazyClassifier

    # Usar amostra menor para evitar OOM em sistemas com pouca RAM
    sample_size = min(5000, len(X_train_balanced))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train_balanced), sample_size, replace=False)
    X_lazy_train = X_train_balanced[idx]
    y_lazy_train = y_train_balanced[idx]

    lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    lazy_models, lazy_predictions = lazy_clf.fit(
        X_lazy_train, X_test_scaled, y_lazy_train, y_test
    )

    print("\n  Resultados LazyPredict (Top 15):")
    print(lazy_models.head(15).to_string())

    # Salvar resultados
    lazy_models.to_csv(os.path.join(FIGURES_DIR, 'lazy_predict_results.csv'))
    print("\n  ✓ Resultados salvos: lazy_predict_results.csv")

    # Gráfico LazyPredict
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    top_models = lazy_models.head(15)

    # Accuracy
    top_models['Accuracy'].sort_values().plot(
        kind='barh', ax=axes[0], color='#2c3e50', edgecolor='white'
    )
    axes[0].set_title('Top 15 Modelos — Accuracy', fontweight='bold')
    axes[0].set_xlabel('Accuracy')

    # F1 Score
    if 'F1 Score' in top_models.columns:
        f1_col = 'F1 Score'
    else:
        f1_col = top_models.columns[1]

    top_models[f1_col].sort_values().plot(
        kind='barh', ax=axes[1], color='#8e44ad', edgecolor='white'
    )
    axes[1].set_title(f'Top 15 Modelos — {f1_col}', fontweight='bold')
    axes[1].set_xlabel(f1_col)

    plt.suptitle('LazyPredict — Comparação Automática de Modelos', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '08_lazy_predict.png'))
    plt.close()
    print("  ✓ Gráfico salvo: 08_lazy_predict.png")

except Exception as e:
    print(f"  ⚠ LazyPredict falhou: {e}")
    print("  Continuando com modelos manuais...")

# 4.2 Treinamento detalhado dos melhores modelos com MLflow
print("\n--- 4.2 Treinamento com MLflow Tracking ---")

# Configurar MLflow
mlflow_uri = os.path.join(BASE_DIR, 'mlruns')
mlflow.set_tracking_uri(f'file://{mlflow_uri}')
experiment_name = "cardiovascular_risk_classifier"

try:
    experiment_id = mlflow.create_experiment(experiment_name)
except:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

mlflow.set_experiment(experiment_name)
print(f"  MLflow tracking URI: file://{mlflow_uri}")
print(f"  Experiment: {experiment_name}")

# Modelos candidatos com hiperparâmetros pré-otimizados
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=2,
        random_state=RANDOM_STATE, n_jobs=2
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE
    ),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=200, max_depth=20, min_samples_split=2,
        random_state=RANDOM_STATE, n_jobs=2
    ),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'KNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'AdaBoost': AdaBoostClassifier(random_state=RANDOM_STATE, algorithm='SAMME'),
}

results = {}
best_score = 0
best_model_name = None
best_model = None

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    print(f"\n  Treinando: {name}...")

    with mlflow.start_run(run_name=name):
        trained_model = model.fit(X_train_balanced, y_train_balanced)
        best_params = trained_model.get_params()
        # Log selective params
        params_to_log = {k: v for k, v in best_params.items()
                         if isinstance(v, (int, float, str, bool)) and v is not None}
        mlflow.log_params(params_to_log)

        # Predição
        y_pred = trained_model.predict(X_test_scaled)

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Cross-validation (3-fold para rapidez)
        cv_scores = cross_val_score(trained_model, X_train_balanced, y_train_balanced,
                                     cv=cv, scoring='f1_weighted', n_jobs=2)

        # ROC AUC
        try:
            if hasattr(trained_model, 'predict_proba'):
                y_proba = trained_model.predict_proba(X_test_scaled)
                y_test_bin = label_binarize(y_test, classes=np.unique(y))
                roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='weighted')
            else:
                roc_auc = 0.0
        except:
            roc_auc = 0.0

        # Log métricas
        metrics = {
            'accuracy': round(acc, 4),
            'precision_weighted': round(prec, 4),
            'recall_weighted': round(rec, 4),
            'f1_weighted': round(f1, 4),
            'roc_auc_weighted': round(roc_auc, 4),
            'cv_f1_mean': round(cv_scores.mean(), 4),
            'cv_f1_std': round(cv_scores.std(), 4)
        }
        mlflow.log_metrics(metrics)

        # Log modelo
        mlflow.sklearn.log_model(trained_model, f"model_{name}")

        results[name] = {
            'model': trained_model,
            'metrics': metrics,
            'y_pred': y_pred,
            'cv_scores': cv_scores
        }

        print(f"    Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f} | CV F1: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")

        if f1 > best_score:
            best_score = f1
            best_model_name = name
            best_model = trained_model

print(f"\n  🏆 Melhor modelo: {best_model_name} (F1={best_score:.4f})")


# =============================================================================
#  FASE 5 — EVALUATION (Avaliação)
# =============================================================================
print("\n" + "=" * 80)
print(" FASE 5 — EVALUATION")
print("=" * 80)

# 5.1 Tabela comparativa
print("\n--- 5.1 Tabela Comparativa dos Modelos ---")
comparison = pd.DataFrame({
    name: r['metrics'] for name, r in results.items()
}).T.sort_values('f1_weighted', ascending=False)
print(comparison.to_string())
comparison.to_csv(os.path.join(FIGURES_DIR, 'comparacao_modelos.csv'))

# Gráfico de comparação
fig, ax = plt.subplots(figsize=(14, 7))
metrics_to_plot = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_weighted']
x = np.arange(len(comparison))
width = 0.15
metric_colors = ['#2c3e50', '#2980b9', '#27ae60', '#e74c3c', '#8e44ad']

for i, (metric, color) in enumerate(zip(metrics_to_plot, metric_colors)):
    ax.bar(x + i * width, comparison[metric], width, label=metric.replace('_', ' ').title(), color=color)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(comparison.index, rotation=30, ha='right')
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Comparação de Modelos — Métricas de Avaliação', fontweight='bold', fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '09_comparacao_modelos.png'))
plt.close()
print("  ✓ Gráfico salvo: 09_comparacao_modelos.png")

# 5.2 Confusion Matrix do melhor modelo
print(f"\n--- 5.2 Confusion Matrix — {best_model_name} ---")
y_pred_best = results[best_model_name]['y_pred']
cm = confusion_matrix(y_test, y_pred_best)
cm_norm = confusion_matrix(y_test, y_pred_best, normalize='true')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=RISK_ORDER, yticklabels=RISK_ORDER)
axes[0].set_title(f'Confusion Matrix — {best_model_name}', fontweight='bold')
axes[0].set_ylabel('Real')
axes[0].set_xlabel('Predito')

sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
            xticklabels=RISK_ORDER, yticklabels=RISK_ORDER)
axes[1].set_title(f'Confusion Matrix Normalizada — {best_model_name}', fontweight='bold')
axes[1].set_ylabel('Real')
axes[1].set_xlabel('Predito')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '10_confusion_matrix.png'))
plt.close()
print("  ✓ Gráfico salvo: 10_confusion_matrix.png")

# 5.3 Classification Report
print(f"\n--- 5.3 Classification Report — {best_model_name} ---")
report = classification_report(y_test, y_pred_best, target_names=RISK_ORDER)
print(report)

report_dict = classification_report(y_test, y_pred_best, target_names=RISK_ORDER, output_dict=True)
with open(os.path.join(FIGURES_DIR, 'classification_report.json'), 'w', encoding='utf-8') as f:
    json.dump(report_dict, f, ensure_ascii=False, indent=2)

# 5.4 Curvas ROC Multiclass
print("\n--- 5.4 Curvas ROC Multiclass ---")
if hasattr(best_model, 'predict_proba'):
    y_proba = best_model.predict_proba(X_test_scaled)
    y_test_bin = label_binarize(y_test, classes=np.arange(len(RISK_ORDER)))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (risk, color) in enumerate(zip(RISK_ORDER, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc_i = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{risk} (AUC = {roc_auc_i:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
    ax.set_title(f'Curvas ROC — {best_model_name}', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '11_roc_curves.png'))
    plt.close()
    print("  ✓ Gráfico salvo: 11_roc_curves.png")

# 5.5 Feature Importance
print(f"\n--- 5.5 Feature Importance — {best_model_name} ---")
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print(feat_imp.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 7))
    feat_imp_sorted = feat_imp.sort_values('Importance', ascending=True)
    colors_imp = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feat_imp_sorted)))
    ax.barh(feat_imp_sorted['Feature'], feat_imp_sorted['Importance'],
            color=colors_imp, edgecolor='white')
    ax.set_xlabel('Importância', fontsize=12)
    ax.set_title(f'Feature Importance — {best_model_name}', fontweight='bold', fontsize=14)
    for i, (imp, name) in enumerate(zip(feat_imp_sorted['Importance'], feat_imp_sorted['Feature'])):
        ax.text(imp + 0.002, i, f'{imp:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '12_feature_importance.png'))
    plt.close()
    print("  ✓ Gráfico salvo: 12_feature_importance.png")

    feat_imp.to_csv(os.path.join(FIGURES_DIR, 'feature_importance.csv'), index=False)

# 5.6 Cross-validation boxplot
print("\n--- 5.6 Cross-Validation Comparison ---")
fig, ax = plt.subplots(figsize=(12, 6))
cv_data = [r['cv_scores'] for r in results.values()]
bp = ax.boxplot(cv_data, labels=list(results.keys()), patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
model_colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
for patch, color in zip(bp['boxes'], model_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_ylabel('F1 Score (weighted)', fontsize=12)
ax.set_title('Cross-Validation (5-Fold) — F1 Score por Modelo', fontweight='bold', fontsize=14)
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '13_cross_validation.png'))
plt.close()
print("  ✓ Gráfico salvo: 13_cross_validation.png")


# =============================================================================
#  FASE 6 — DEPLOYMENT (Implantação)
# =============================================================================
print("\n" + "=" * 80)
print(" FASE 6 — DEPLOYMENT")
print("=" * 80)

# 6.1 Salvar artefatos
print("\n--- 6.1 Salvando Artefatos do Modelo ---")

# Modelo
model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
joblib.dump(best_model, model_path)
print(f"  ✓ Modelo salvo: {model_path}")

# Scaler
scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"  ✓ Scaler salvo: {scaler_path}")

# Label encoder
encoder_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
joblib.dump(le_target, encoder_path)
print(f"  ✓ Label encoder salvo: {encoder_path}")

# Metadata
metadata = {
    'best_model_name': best_model_name,
    'feature_columns': feature_cols,
    'target_classes': RISK_ORDER,
    'metrics': results[best_model_name]['metrics'],
    'encoding_info': {
        'Sexo': 'Sexo_Masculino (1=Masculino, 0=Feminino)',
        'Fumante': 'Fumante_Sim (1=Sim, 0=Não)',
        'Alcool': 'Alcool_Encoded (0=Baixo, 1=Moderado, 2=Alto)',
        'Historico_Familiar': 'Historico_Familiar_Sim (1=Sim, 0=Não)'
    },
    'num_features_original': num_cols,
    'alcool_map': alcool_map
}

metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"  ✓ Metadata salvo: {metadata_path}")

# Classification report
report_path = os.path.join(MODELS_DIR, 'classification_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report_dict, f, ensure_ascii=False, indent=2)
print(f"  ✓ Classification report salvo: {report_path}")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    feat_imp_path = os.path.join(MODELS_DIR, 'feature_importance.csv')
    feat_imp.to_csv(feat_imp_path, index=False)
    print(f"  ✓ Feature importance salvo: {feat_imp_path}")


print("\n" + "=" * 80)
print(" PIPELINE CRISP-DM CONCLUÍDO COM SUCESSO!")
print("=" * 80)
print(f"""
  📊 Gráficos salvos em: {FIGURES_DIR}
  🤖 Modelo salvo em: {MODELS_DIR}
  📈 MLflow tracking em: file://{mlflow_uri}
  
  Para visualizar MLflow UI:
    mlflow ui --backend-store-uri file://{mlflow_uri} --port 5000
    
  🏆 Melhor modelo: {best_model_name}
  📋 Métricas: {results[best_model_name]['metrics']}
  
  ⚠ DISCLAIMER: Este conteúdo é destinado apenas para fins educacionais.
  Os dados exibidos são ilustrativos e podem não corresponder a situações reais.
""")
