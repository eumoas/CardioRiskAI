# 🫀 CardioRisk AI — Classificador de Risco Cardiovascular

> Projeto final da UC **Aprendizado de Máquina** — 2026/1  
> Classificação de risco cardiovascular baseada nas diretrizes da **Sociedade Brasileira de Cardiologia (SBC)**

---

## 📋 Descrição

O **CardioRisk AI** é um sistema de classificação de risco cardiovascular que utiliza técnicas de Machine Learning para estratificar pacientes em quatro níveis de risco, seguindo o **Escore de Risco Global de Framingham** adaptado pela SBC:

| Nível | Probabilidade de eventos em 10 anos |
|-------|-------------------------------------|
| 💚 **Baixo** | < 5% |
| 💛 **Moderado** | 5–10% (mulheres) / 5–20% (homens) |
| ❤️‍🔥 **Alto** | > 10% (mulheres) / > 20% (homens) |
| 🟣 **Muito Alto** | Eventos prévios / condições graves |

---

## 🗂️ Estrutura do Projeto

```
├── app/
│   └── app.py                  # Aplicação Streamlit (interface principal)
├── notebooks/
│   └── analise_crisp_dm.py     # Pipeline completo CRISP-DM
├── models/                     # Artefatos do modelo treinado (gerados pelo pipeline)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── model_metadata.json
│   ├── classification_report.json
│   └── feature_importance.csv
├── data/
│   └── dataset_saude_brasil.csv  # Dataset (20.000 registros)
├── db/
│   └── init.sql                # Script de inicialização do PostgreSQL
├── .streamlit/
│   └── config.toml             # Configurações do Streamlit
├── Dockerfile                  # Container da aplicação Streamlit
├── Dockerfile.mlflow           # Container do servidor MLflow
├── docker-compose.yml          # Orquestração dos containers
├── requirements.txt            # Dependências Python
└── README.md
```

---

## 🚀 Tecnologias Utilizadas

| Camada | Tecnologia |
|--------|------------|
| **Interface** | [Streamlit](https://streamlit.io/) |
| **ML / Modelagem** | Scikit-learn, LazyPredict, SMOTE (imbalanced-learn) |
| **Rastreamento de Experimentos** | [MLflow](https://mlflow.org/) |
| **Banco de Dados** | PostgreSQL 15 |
| **Containerização** | Docker + Docker Compose |
| **Visualização** | Plotly, Matplotlib, Seaborn |

---

## 🧠 Metodologia CRISP-DM

O pipeline (`notebooks/analise_crisp_dm.py`) segue as seis fases da metodologia **CRISP-DM**:

1. **Business Understanding** — Definição do problema clínico e objetivos  
2. **Data Understanding** — EDA completa com testes estatísticos (Kruskal-Wallis, Qui-Quadrado, D'Agostino-Pearson), análise de correlações (Pearson e Spearman) e detecção de outliers (IQR)  
3. **Data Preparation** — Tratamento de nulos, encoding de variáveis categóricas, normalização (StandardScaler) e balanceamento de classes com **SMOTE**  
4. **Modeling** — Comparação automática com **LazyPredict** + treinamento detalhado de 7 modelos (Random Forest, Gradient Boosting, Extra Trees, Logistic Regression, KNN, Decision Tree, AdaBoost) com rastreamento via **MLflow**  
5. **Evaluation** — F1-Score ponderado, ROC-AUC multiclasse, Confusion Matrix, Curvas ROC por classe e Feature Importance  
6. **Deployment** — Interface Streamlit containerizada com Docker, persistência em PostgreSQL  

---

## ▶️ Como Executar

### Opção 1 — Docker Compose (Recomendado)

```bash
# Clonar o repositório
git clone https://github.com/SEU_USUARIO/cardiorisk-ai.git
cd cardiorisk-ai

# Gerar os modelos (necessário na primeira execução)
pip install -r requirements.txt
python notebooks/analise_crisp_dm.py

# Subir todos os serviços
docker compose up --build
```

Serviços disponíveis:
- **Aplicação:** http://localhost:8501  
- **MLflow UI:** http://localhost:5000  
- **PostgreSQL:** porta 5432  

---

### Opção 2 — Execução Local (sem Docker)

```bash
# Instalar dependências
pip install -r requirements.txt

# Gerar modelos (se ainda não existirem)
python notebooks/analise_crisp_dm.py

# Iniciar a aplicação
streamlit run app/app.py
```

---

## 📊 Features Utilizadas pelo Modelo

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `Idade` | Numérica | Idade em anos |
| `IMC` | Numérica | Calculado automaticamente (peso/altura²) |
| `Pressao_Sistolica` | Numérica | mmHg |
| `Pressao_Diastolica` | Numérica | mmHg |
| `Colesterol` | Numérica | mg/dL |
| `Frequencia_Cardiaca_Repouso` | Numérica | bpm |
| `Passos_Diarios` | Numérica | Passos por dia |
| `Horas_Sono` | Numérica | Horas/noite |
| `Agua_Litros` | Numérica | Litros/dia |
| `Calorias` | Numérica | kcal/dia |
| `Horas_Trabalho` | Numérica | Horas/dia |
| `Sexo` | Categórica | Masculino / Feminino |
| `Fumante` | Categórica | Sim / Não |
| `Alcool` | Categórica | Baixo / Moderado / Alto |
| `Historico_Familiar` | Categórica | Sim / Não |

---

## 🗄️ Banco de Dados

A aplicação persiste automaticamente os dados de cada classificação realizada em uma tabela PostgreSQL (`pacientes`), armazenando:
- Dados pessoais e clínicos do paciente
- IMC calculado, peso e altura
- Nível de risco classificado e probabilidades por classe
- Timestamp do registro

> ⚠️ **Sem banco de dados:** a classificação continua funcionando normalmente; apenas a persistência é desativada.

---

## 📁 Dataset

- **Nome:** Dataset Saúde Brasil  
- **Registros:** 20.000 (dados simulados para fins educacionais)  
- **Referência clínica:** SBC — Escore de Risco Global de Framingham  

> Este projeto e seus dados têm **fins exclusivamente educacionais**.

---

## 👤 Autor

Desenvolvido como **Desafio Final** da UC de Aprendizado de Máquina — 2026/1.
