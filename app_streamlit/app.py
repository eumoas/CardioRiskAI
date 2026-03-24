#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
  CLASSIFICADOR DE RISCO CARDIOVASCULAR — Aplicação Streamlit
  Interface profissional com estética médica premium
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
#  CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="CardioRisk AI — Classificador de Risco Cardiovascular",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
#  PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# =============================================================================
#  CSS PREMIUM — TEMA MÉDICO DARK
# =============================================================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Root variables */
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: #1a1f2e;
        --text-primary: #f0f4f8;
        --text-secondary: #94a3b8;
        --accent-blue: #3b82f6;
        --accent-cyan: #06b6d4;
        --risk-baixo: #10b981;
        --risk-moderado: #f59e0b;
        --risk-alto: #ef4444;
        --risk-muito-alto: #a855f7;
        --border-color: rgba(255,255,255,0.06);
        --glass-bg: rgba(26, 31, 46, 0.7);
    }

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1729 0%, #111827 100%);
        border-right: 1px solid var(--border-color);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f0f4f8;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(26,31,46,0.9) 0%, rgba(17,24,39,0.9) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(59,130,246,0.15);
    }

    /* Risk result card */
    .risk-card {
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        backdrop-filter: blur(12px);
        border: 2px solid;
        animation: fadeInScale 0.6s ease-out;
    }
    @keyframes fadeInScale {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }

    .risk-baixo {
        background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(6,182,212,0.1) 100%);
        border-color: rgba(16,185,129,0.4);
    }
    .risk-moderado {
        background: linear-gradient(135deg, rgba(245,158,11,0.15) 0%, rgba(251,191,36,0.1) 100%);
        border-color: rgba(245,158,11,0.4);
    }
    .risk-alto {
        background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(248,113,113,0.1) 100%);
        border-color: rgba(239,68,68,0.4);
    }
    .risk-muito-alto {
        background: linear-gradient(135deg, rgba(168,85,247,0.15) 0%, rgba(192,132,252,0.1) 100%);
        border-color: rgba(168,85,247,0.4);
    }

    .risk-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .risk-subtitle {
        font-size: 1.1rem;
        opacity: 0.85;
        font-weight: 400;
    }

    /* Header styles */
    .app-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    .app-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6, #06b6d4, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
        margin-bottom: 0.3rem;
    }
    .app-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Metric mini */
    .metric-mini {
        text-align: center;
        padding: 1rem;
    }
    .metric-mini .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3b82f6;
    }
    .metric-mini .label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }

    /* Probability bars */
    .prob-container {
        margin: 0.6rem 0;
    }
    .prob-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
        font-size: 0.9rem;
    }
    .prob-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 1s ease-out;
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.2);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-size: 0.85rem;
        color: #fbbf24;
        margin-top: 2rem;
    }

    /* Divider custom */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(59,130,246,0.3), transparent);
        margin: 2rem 0;
    }

    /* Sidebar form styling */
    section[data-testid="stSidebar"] .stSlider > div > div > div {
        color: #3b82f6;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(59,130,246,0.35) !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  CARREGAMENTO DOS ARTEFATOS
# =============================================================================
@st.cache_resource
def load_model():
    """Carrega o modelo, scaler, encoder e metadata."""
    model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))

    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    with open(os.path.join(MODELS_DIR, 'classification_report.json'), 'r', encoding='utf-8') as f:
        report = json.load(f)

    feat_imp_path = os.path.join(MODELS_DIR, 'feature_importance.csv')
    feat_imp = pd.read_csv(feat_imp_path) if os.path.exists(feat_imp_path) else None

    return model, scaler, label_encoder, metadata, report, feat_imp


model, scaler, label_encoder, metadata, report, feat_imp = load_model()

RISK_ORDER = metadata['target_classes']  # ['Baixo', 'Moderado', 'Alto', 'Muito Alto']
RISK_COLORS = {
    'Baixo': '#10b981',
    'Moderado': '#f59e0b',
    'Alto': '#ef4444',
    'Muito Alto': '#a855f7'
}
RISK_ICONS = {
    'Baixo': '💚',
    'Moderado': '💛',
    'Alto': '❤️‍🔥',
    'Muito Alto': '🟣'
}
RISK_CSS_CLASS = {
    'Baixo': 'risk-baixo',
    'Moderado': 'risk-moderado',
    'Alto': 'risk-alto',
    'Muito Alto': 'risk-muito-alto'
}
RISK_TEXT_COLOR = {
    'Baixo': '#10b981',
    'Moderado': '#f59e0b',
    'Alto': '#ef4444',
    'Muito Alto': '#a855f7'
}
RISK_DESCRIPTIONS = {
    'Baixo': 'Probabilidade de eventos cardiovasculares < 5% em 10 anos. Manter hábitos saudáveis.',
    'Moderado': 'Probabilidade de 5-10% (mulheres) ou 5-20% (homens) em 10 anos. Acompanhamento médico recomendado.',
    'Alto': 'Probabilidade > 10% (mulheres) ou > 20% (homens) em 10 anos. Intervenção médica necessária.',
    'Muito Alto': 'Presença de eventos cardiovasculares prévios ou condições graves. Tratamento intensivo recomendado.'
}


# =============================================================================
#  HEADER
# =============================================================================
st.markdown("""
<div class="app-header">
    <h1>🫀 CardioRisk AI</h1>
    <p>Classificador Inteligente de Risco Cardiovascular — Baseado nas Diretrizes da SBC</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# =============================================================================
#  SIDEBAR — FORMULÁRIO DE ENTRADA
# =============================================================================
with st.sidebar:
    st.markdown("## 📋 Dados do Paciente")
    st.markdown("---")

    st.markdown("### 👤 Informações Pessoais")
    idade = st.slider("Idade (anos)", 18, 90, 45, help="Idade do paciente em anos")
    sexo = st.selectbox("Sexo", ["Feminino", "Masculino"])

    st.markdown("### 📏 Medidas Corporais")
    imc = st.slider("IMC (kg/m²)", 15.0, 50.0, 25.0, 0.1,
                     help="Índice de Massa Corporal")

    st.markdown("### 🏃 Estilo de Vida")
    passos_diarios = st.slider("Passos Diários", 0, 30000, 7000, 100,
                                help="Média de passos por dia")
    horas_sono = st.slider("Horas de Sono", 3.0, 12.0, 7.0, 0.5)
    agua_litros = st.slider("Água (litros/dia)", 0.5, 5.0, 2.0, 0.1)
    calorias = st.slider("Calorias (kcal/dia)", 1000, 5000, 2200, 50)
    horas_trabalho = st.slider("Horas de Trabalho/dia", 0, 16, 8)

    st.markdown("### 🩺 Dados Clínicos")
    freq_cardiaca = st.slider("Frequência Cardíaca em Repouso (bpm)", 40, 120, 72)
    pressao_sistolica = st.slider("Pressão Sistólica (mmHg)", 80, 200, 120)
    pressao_diastolica = st.slider("Pressão Diastólica (mmHg)", 50, 130, 80)
    colesterol = st.slider("Colesterol Total (mg/dL)", 100, 400, 200)

    st.markdown("### 🚬 Hábitos e Histórico")
    fumante = st.selectbox("Fumante", ["Não", "Sim"])
    alcool = st.selectbox("Consumo de Álcool", ["Baixo", "Moderado", "Alto"])
    historico_familiar = st.selectbox("Histórico Familiar de DCV", ["Não", "Sim"])

    st.markdown("---")
    predict_button = st.button("🔬 Classificar Risco", use_container_width=True)


# =============================================================================
#  FUNÇÃO DE PREDIÇÃO
# =============================================================================
def make_prediction(input_data: dict) -> tuple:
    """Realiza a predição de risco cardiovascular."""
    alcool_map = metadata.get('alcool_map', {'Baixo': 0, 'Moderado': 1, 'Alto': 2})

    features = np.array([[
        input_data['Idade'],
        input_data['IMC'],
        input_data['Passos_Diarios'],
        input_data['Horas_Sono'],
        input_data['Agua_Litros'],
        input_data['Calorias'],
        input_data['Horas_Trabalho'],
        input_data['Frequencia_Cardiaca_Repouso'],
        input_data['Pressao_Sistolica'],
        input_data['Pressao_Diastolica'],
        input_data['Colesterol'],
        1 if input_data['Sexo'] == 'Masculino' else 0,
        1 if input_data['Fumante'] == 'Sim' else 0,
        alcool_map.get(input_data['Alcool'], 0),
        1 if input_data['Historico_Familiar'] == 'Sim' else 0
    ]])

    features_scaled = scaler.transform(features)
    prediction_encoded = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    return prediction_label, probabilities


# =============================================================================
#  RESULTADO DA PREDIÇÃO
# =============================================================================
if predict_button:
    input_data = {
        'Idade': idade,
        'IMC': imc,
        'Passos_Diarios': passos_diarios,
        'Horas_Sono': horas_sono,
        'Agua_Litros': agua_litros,
        'Calorias': calorias,
        'Horas_Trabalho': horas_trabalho,
        'Frequencia_Cardiaca_Repouso': freq_cardiaca,
        'Pressao_Sistolica': pressao_sistolica,
        'Pressao_Diastolica': pressao_diastolica,
        'Colesterol': colesterol,
        'Sexo': sexo,
        'Fumante': fumante,
        'Alcool': alcool,
        'Historico_Familiar': historico_familiar
    }

    risk_label, probabilities = make_prediction(input_data)
    risk_color = RISK_TEXT_COLOR[risk_label]
    risk_icon = RISK_ICONS[risk_label]
    risk_css = RISK_CSS_CLASS[risk_label]
    risk_desc = RISK_DESCRIPTIONS[risk_label]

    # ---- Resultado principal ----
    st.markdown(f"""
    <div class="risk-card {risk_css}">
        <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">{risk_icon}</div>
        <div class="risk-title" style="color: {risk_color};">Risco {risk_label}</div>
        <div class="risk-subtitle" style="color: {risk_color};">{risk_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- Probabilidades por classe ----
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📊 Probabilidade por Nível de Risco")

        # model.classes_ são os valores encoded (0,1,2,3)
        # label_encoder.inverse_transform converte de volta para strings
        prob_mapped = {}
        for i, encoded_class in enumerate(model.classes_):
            decoded = label_encoder.inverse_transform([encoded_class])[0]
            prob_mapped[decoded] = probabilities[i]

        for risk in RISK_ORDER:
            prob = prob_mapped.get(risk, 0.0)
            color = RISK_COLORS[risk]
            pct = prob * 100
            st.markdown(f"""
            <div class="prob-container">
                <div class="prob-label">
                    <span style="color: {color}; font-weight: 600;">{risk}</span>
                    <span style="color: {color}; font-weight: 700;">{pct:.1f}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: {pct}%; background: linear-gradient(90deg, {color}, {color}aa);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎯 Distribuição de Probabilidade")

        fig_pie = go.Figure(data=[go.Pie(
            labels=[r for r in RISK_ORDER],
            values=[prob_mapped.get(r, 0.0) * 100 for r in RISK_ORDER],
            hole=0.55,
            marker=dict(colors=[RISK_COLORS[r] for r in RISK_ORDER]),
            textinfo='label+percent',
            textfont=dict(size=12, color='white'),
            hovertemplate='%{label}: %{value:.1f}%<extra></extra>'
        )])
        fig_pie.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            annotations=[dict(
                text=f'<b>{prob_mapped.get(risk_label, 0.0)*100:.0f}%</b>',
                x=0.5, y=0.5, font_size=28, font_color=risk_color,
                showarrow=False
            )]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- Gauge radar dos fatores de risco ----
    st.markdown("### 🩺 Perfil do Paciente — Fatores de Risco")

    # Normalizar os inputs para um radar chart (0-100 scale)
    radar_data = {
        'Idade': min(100, (idade - 18) / (90 - 18) * 100),
        'IMC': min(100, (imc - 15) / (50 - 15) * 100),
        'Pressão Sistólica': min(100, (pressao_sistolica - 80) / (200 - 80) * 100),
        'Colesterol': min(100, (colesterol - 100) / (400 - 100) * 100),
        'Freq. Cardíaca': min(100, (freq_cardiaca - 40) / (120 - 40) * 100),
        'Tabagismo': 100 if fumante == 'Sim' else 0,
        'Álcool': {'Baixo': 15, 'Moderado': 55, 'Alto': 95}.get(alcool, 0),
        'Hist. Familiar': 100 if historico_familiar == 'Sim' else 0,
    }

    categories = list(radar_data.keys())
    values = list(radar_data.values())
    values.append(values[0])  # close the polygon

    fig_radar = go.Figure()
    # Converter hex para rgba para compatibilidade com Plotly
    rc = risk_color.lstrip('#')
    fill_rgba = f'rgba({int(rc[0:2],16)}, {int(rc[2:4],16)}, {int(rc[4:6],16)}, 0.13)'

    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor=fill_rgba,
        line=dict(color=risk_color, width=2),
        marker=dict(size=6, color=risk_color),
        name='Paciente'
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor='rgba(255,255,255,0.08)',
                tickfont=dict(color='#64748b', size=9)
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.08)',
                tickfont=dict(color='#94a3b8', size=11)
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(t=30, b=30, l=60, r=60),
        showlegend=False
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# =============================================================================
#  TABS — INFORMAÇÕES DO MODELO
# =============================================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Métricas do Modelo", "🔬 Feature Importance", "ℹ️ Sobre"])

# ---- Tab 1: Métricas ----
with tab1:
    st.markdown("### 🏆 Performance do Modelo")
    st.markdown(f"**Modelo selecionado:** `{metadata['best_model_name']}`")

    metrics = metadata['metrics']

    col1, col2, col3, col4, col5 = st.columns(5)

    metric_items = [
        ("Accuracy", metrics['accuracy'], col1),
        ("Precision", metrics['precision_weighted'], col2),
        ("Recall", metrics['recall_weighted'], col3),
        ("F1 Score", metrics['f1_weighted'], col4),
        ("ROC AUC", metrics['roc_auc_weighted'], col5),
    ]

    for label, value, col in metric_items:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-mini">
                    <div class="value">{value:.1%}</div>
                    <div class="label">{label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown(f"**Cross-Validation (F1):** {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")

    # Classification report por classe
    st.markdown("### 📋 Relatório por Classe de Risco")

    report_rows = []
    for risk in RISK_ORDER:
        if risk in report:
            r = report[risk]
            report_rows.append({
                'Classe': risk,
                'Precision': f"{r['precision']:.3f}",
                'Recall': f"{r['recall']:.3f}",
                'F1-Score': f"{r['f1-score']:.3f}",
                'Suporte': int(r['support'])
            })

    if report_rows:
        st.dataframe(
            pd.DataFrame(report_rows).set_index('Classe'),
            use_container_width=True
        )

    # Gráfico de barras das métricas por classe
    if report_rows:
        df_report = pd.DataFrame(report_rows)
        fig_bar = go.Figure()
        for metric_name in ['Precision', 'Recall', 'F1-Score']:
            fig_bar.add_trace(go.Bar(
                name=metric_name,
                x=df_report['Classe'],
                y=[float(v) for v in df_report[metric_name]],
                marker_color={'Precision': '#3b82f6', 'Recall': '#06b6d4', 'F1-Score': '#10b981'}[metric_name]
            ))
        fig_bar.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.06)',
                range=[0, 1.05],
                tickfont=dict(color='#94a3b8')
            ),
            xaxis=dict(tickfont=dict(color='#94a3b8')),
            legend=dict(font=dict(color='#94a3b8')),
            margin=dict(t=20, b=40)
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ---- Tab 2: Feature Importance ----
with tab2:
    st.markdown("### 🔬 Importância das Features")

    if feat_imp is not None:
        feat_sorted = feat_imp.sort_values('Importance', ascending=True)

        # Map feature names to Portuguese labels
        feature_labels = {
            'Colesterol': '🧪 Colesterol',
            'Idade': '🎂 Idade',
            'Fumante_Sim': '🚬 Tabagismo',
            'IMC': '📏 IMC',
            'Alcool_Encoded': '🍷 Álcool',
            'Horas_Trabalho': '💼 Horas Trabalho',
            'Historico_Familiar_Sim': '🧬 Hist. Familiar',
            'Horas_Sono': '😴 Horas Sono',
            'Passos_Diarios': '🏃 Passos Diários',
            'Calorias': '🔥 Calorias',
            'Agua_Litros': '💧 Água',
            'Pressao_Diastolica': '🩸 P. Diastólica',
            'Pressao_Sistolica': '🩸 P. Sistólica',
            'Frequencia_Cardiaca_Repouso': '❤️ Freq. Cardíaca',
            'Sexo_Masculino': '♂️ Sexo'
        }

        labels = [feature_labels.get(f, f) for f in feat_sorted['Feature']]

        # Color gradient
        n_features = len(feat_sorted)
        colors = [f'rgba({int(239 - i * (239-16)/n_features)}, {int(68 + i * (185-68)/n_features)}, {int(68 + i * (129-68)/n_features)}, 0.85)' for i in range(n_features)]

        fig_imp = go.Figure(go.Bar(
            x=feat_sorted['Importance'],
            y=labels,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(width=0)
            ),
            text=[f'{v:.4f}' for v in feat_sorted['Importance']],
            textposition='outside',
            textfont=dict(color='#94a3b8', size=11)
        ))
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.06)',
                tickfont=dict(color='#94a3b8'),
                title=dict(text='Importância', font=dict(color='#94a3b8'))
            ),
            yaxis=dict(tickfont=dict(color='#e2e8f0', size=12)),
            margin=dict(t=20, b=40, l=150)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""
        > **Interpretação:** O Colesterol, a Idade e o Tabagismo são os fatores mais determinantes
        > na classificação de risco cardiovascular, seguidos pelo IMC e consumo de Álcool.
        """)
    else:
        st.info("Feature importance não disponível para este modelo.")


# ---- Tab 3: Sobre ----
with tab3:
    st.markdown("### ℹ️ Sobre o CardioRisk AI")

    st.markdown("""
    #### Metodologia

    Este classificador foi desenvolvido seguindo a metodologia **CRISP-DM** (Cross-Industry Standard
    Process for Data Mining) e é baseado nos critérios de estratificação de risco da **Sociedade
    Brasileira de Cardiologia (SBC)**, utilizando o Escore de Risco Global de Framingham.

    #### Níveis de Risco

    | Nível | Critério |
    |:------|:---------|
    | 🟢 **Baixo** | Probabilidade < 5% de eventos em 10 anos |
    | 🟡 **Moderado** | Probabilidade 5-10% (♀) / 5-20% (♂) |
    | 🔴 **Alto** | Probabilidade > 10% (♀) / > 20% (♂) |
    | 🟣 **Muito Alto** | Eventos prévios / condições graves |

    #### Fatores Considerados

    - **Demográficos:** Idade, Sexo
    - **Antropométricos:** IMC
    - **Clínicos:** PA Sistólica/Diastólica, Colesterol, Freq. Cardíaca
    - **Estilo de vida:** Passos diários, Sono, Hidratação, Calorias, Horas de trabalho
    - **Hábitos:** Tabagismo, Álcool
    - **Genéticos:** Histórico Familiar de DCV

    #### Tecnologias

    | Componente | Tecnologia |
    |:-----------|:-----------|
    | Modelo | GradientBoosting (scikit-learn) |
    | Interface | Streamlit |
    | Tracking | MLflow |
    | Container | Docker |
    """)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>DISCLAIMER:</strong> Este conteúdo é destinado apenas para fins educacionais.
        Os dados exibidos são ilustrativos e podem não corresponder a situações reais.
        Não substitui avaliação médica profissional. Consulte sempre um cardiologista
        para avaliação de risco cardiovascular.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
#  FOOTER
# =============================================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #475569; font-size: 0.8rem;">
    CardioRisk AI — Desafio Final de Aprendizado de Máquina<br>
    Desenvolvido com Streamlit • MLflow • Docker • scikit-learn
</div>
""", unsafe_allow_html=True)
