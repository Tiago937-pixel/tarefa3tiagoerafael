import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                           recall_score, f1_score, confusion_matrix, roc_curve,
                           classification_report)
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="🏨 Análise de Cancelamento de Reservas Hoteleiras",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1e3c72;
    }
    .insight-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🏨 Sistema Inteligente de Predição de Cancelamentos</h1>
    <h3>Análise Avançada com Regressão Logística | Professor João Gabriel</h3>
    <p>Universidade de Brasília - Engenharia de Produção</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Carrega e processa os dados do hotel"""
    try:
        # Tentativa de carregar de diferentes localizações
        possible_paths = [
            'hotel_bookings.csv',
            'data/hotel_bookings.csv', 
            'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv'
        ]
        
        for path in possible_paths:
            try:
                data = pd.read_csv(path)
                if 'is_canceled' in data.columns:
                    return data
            except:
                continue
                
        # Se não conseguir carregar, criar dados sintéticos para demonstração
        st.warning("⚠️ Arquivo de dados não encontrado. Gerando dados sintéticos para demonstração.")
        return generate_synthetic_data()
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return generate_synthetic_data()

def generate_synthetic_data():
    """Gera dados sintéticos realistas para demonstração"""
    np.random.seed(42)
    n = 10000
    
    # Variáveis categóricas
    hotels = np.random.choice(['Resort Hotel', 'City Hotel'], n, p=[0.3, 0.7])
    arrival_months = np.random.choice(['January', 'February', 'March', 'April', 'May', 
                                     'June', 'July', 'August', 'September', 'October', 
                                     'November', 'December'], n)
    customer_types = np.random.choice(['Transient', 'Contract', 'Group', 'Transient-Party'], 
                                    n, p=[0.7, 0.1, 0.1, 0.1])
    market_segments = np.random.choice(['Online TA', 'Direct', 'Corporate', 'Groups', 'Offline TA'], 
                                     n, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    meal_plans = np.random.choice(['BB', 'HB', 'FB', 'SC'], n, p=[0.6, 0.2, 0.1, 0.1])
    
    # Variáveis numéricas
    lead_time = np.random.exponential(30, n).astype(int)
    stays_in_weekend_nights = np.random.poisson(1, n)
    stays_in_week_nights = np.random.poisson(2, n)
    adults = np.random.choice([1, 2, 3, 4], n, p=[0.2, 0.6, 0.15, 0.05])
    children = np.random.choice([0, 1, 2], n, p=[0.8, 0.15, 0.05])
    babies = np.random.choice([0, 1], n, p=[0.95, 0.05])
    adr = np.random.gamma(2, 50, n)
    total_of_special_requests = np.random.poisson(0.5, n)
    booking_changes = np.random.poisson(0.2, n)
    previous_cancellations = np.random.poisson(0.1, n)
    required_car_parking_spaces = np.random.choice([0, 1], n, p=[0.9, 0.1])
    
    # Criar variável target com lógica realística
    prob_cancel = (
        0.1 +  # baseline
        0.3 * (lead_time > 60) +  # lead time alto
        0.2 * (adr > 150) +  # preço alto
        0.15 * (booking_changes > 0) +  # mudanças na reserva
        0.25 * (previous_cancellations > 0) +  # histórico de cancelamento
        0.1 * (customer_types == 'Transient') +  # tipo de cliente
        -0.1 * (total_of_special_requests > 0)  # solicitações especiais reduzem cancelamento
    )
    prob_cancel = np.clip(prob_cancel, 0, 0.8)
    is_canceled = np.random.binomial(1, prob_cancel, n)
    
    # Criar DataFrame
    data = pd.DataFrame({
        'hotel': hotels,
        'is_canceled': is_canceled,
        'lead_time': lead_time,
        'arrival_date_month': arrival_months,
        'stays_in_weekend_nights': stays_in_weekend_nights,
        'stays_in_week_nights': stays_in_week_nights,
        'adults': adults,
        'children': children,
        'babies': babies,
        'meal': meal_plans,
        'market_segment': market_segments,
        'customer_type': customer_types,
        'adr': adr,
        'total_of_special_requests': total_of_special_requests,
        'booking_changes': booking_changes,
        'previous_cancellations': previous_cancellations,
        'required_car_parking_spaces': required_car_parking_spaces,
        'is_repeated_guest': np.random.choice([0, 1], n, p=[0.9, 0.1])
    })
    
    return data

def prepare_data(data):
    """Prepara os dados para modelagem"""
    # Criar cópias para não modificar os dados originais
    df = data.copy()
    
    # Encoding de variáveis categóricas
    categorical_cols = ['hotel', 'arrival_date_month', 'meal', 'market_segment', 'customer_type']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df_encoded

def calculate_vif(X):
    """Calcula o Variance Inflation Factor para detectar multicolinearidade"""
    vif_data = pd.DataFrame()
    vif_data["Variável"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data.sort_values('VIF', ascending=False)

# Carregar dados
data = load_data()

# Sidebar para navegação
st.sidebar.title("📋 Navegação")
page = st.sidebar.selectbox(
    "Escolha a análise:",
    ["🏠 Visão Geral", "📊 Análise Exploratória", "🤖 Modelagem Preditiva", 
     "🔍 Validação de Pressupostos", "💼 Recomendações Estratégicas", "🎯 Simulador de Cenários"]
)

if page == "🏠 Visão Geral":
    st.header("📈 Visão Geral do Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>{:,}</h3>
            <p>Total de Reservas</p>
        </div>
        """.format(len(data)), unsafe_allow_html=True)
    
    with col2:
        cancel_rate = data['is_canceled'].mean()
        st.markdown("""
        <div class="metric-container">
            <h3>{:.1f}%</h3>
            <p>Taxa de Cancelamento</p>
        </div>
        """.format(cancel_rate * 100), unsafe_allow_html=True)
    
    with col3:
        avg_adr = data['adr'].mean() if 'adr' in data.columns else 0
        st.markdown("""
        <div class="metric-container">
            <h3>${:.0f}</h3>
            <p>ADR Médio</p>
        </div>
        """.format(avg_adr), unsafe_allow_html=True)
    
    with col4:
        avg_lead_time = data['lead_time'].mean() if 'lead_time' in data.columns else 0
        st.markdown("""
        <div class="metric-container">
            <h3>{:.0f} dias</h3>
            <p>Lead Time Médio</p>
        </div>
        """.format(avg_lead_time), unsafe_allow_html=True)
    
    st.subheader("📋 Estrutura dos Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Informações do Dataset:**")
        st.write(f"• Linhas: {data.shape[0]:,}")
        st.write(f"• Colunas: {data.shape[1]}")
        st.write(f"• Valores faltantes: {data.isnull().sum().sum()}")
        
    with col2:
        st.write("**Distribuição da Variável Target:**")
        target_dist = data['is_canceled'].value_counts()
        st.write(f"• Não canceladas: {target_dist[0]:,} ({target_dist[0]/len(data)*100:.1f}%)")
        st.write(f"• Canceladas: {target_dist[1]:,} ({target_dist[1]/len(data)*100:.1f}%)")
    
    # Visualização da distribuição
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Distribuição de Cancelamentos", "Cancelamentos por Tipo de Hotel"],
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Gráfico de pizza
    labels = ['Não Cancelada', 'Cancelada']
    values = data['is_canceled'].value_counts()
    fig.add_trace(
        go.Pie(labels=labels, values=values, hole=0.4, name=""),
        row=1, col=1
    )
    
    # Gráfico de barras por hotel
    if 'hotel' in data.columns:
        hotel_cancel = data.groupby('hotel')['is_canceled'].agg(['count', 'mean']).reset_index()
        fig.add_trace(
            go.Bar(x=hotel_cancel['hotel'], y=hotel_cancel['mean']*100, name="Taxa de Cancelamento (%)"),
            row=1, col=2
        )
    
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Preview dos dados
    st.subheader("🔍 Preview dos Dados")
    st.dataframe(data.head(10))

elif page == "📊 Análise Exploratória":
    st.header("📊 Análise Exploratória dos Dados")
    
    tab1, tab2, tab3 = st.tabs(["🔢 Variáveis Numéricas", "📝 Variáveis Categóricas", "🔗 Correlações"])
    
    with tab1:
        st.subheader("Análise de Variáveis Numéricas")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_canceled' in numeric_cols:
            numeric_cols.remove('is_canceled')
        
        selected_numeric = st.multiselect(
            "Selecione as variáveis numéricas para análise:",
            numeric_cols,
            default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
        )
        
        if selected_numeric:
            # Estatísticas descritivas
            st.write("**Estatísticas Descritivas:**")
            st.dataframe(data[selected_numeric].describe())
            
            # Distribuições
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=selected_numeric[:4],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            for i, col in enumerate(selected_numeric[:4]):
                row = i // 2 + 1
                col_pos = i % 2 + 1
                
                fig.add_trace(
                    go.Histogram(x=data[col], name=col, opacity=0.7),
                    row=row, col=col_pos
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Boxplots por cancelamento
            st.subheader("📦 Distribuição por Status de Cancelamento")
            
            selected_for_box = st.selectbox("Escolha uma variável para análise detalhada:", selected_numeric)
            
            fig = px.box(data, x='is_canceled', y=selected_for_box, 
                        title=f"Distribuição de {selected_for_box} por Status de Cancelamento",
                        labels={'is_canceled': 'Cancelado (0=Não, 1=Sim)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Análise de Variáveis Categóricas")
        
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            selected_cat = st.selectbox("Selecione uma variável categórica:", categorical_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribuição geral
                value_counts = data[selected_cat].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Distribuição de {selected_cat}")
                st.plotly_chart(fig)
            
            with col2:
                # Taxa de cancelamento por categoria
                cancel_by_cat = data.groupby(selected_cat)['is_canceled'].agg(['count', 'mean']).reset_index()
                cancel_by_cat['cancel_rate'] = cancel_by_cat['mean'] * 100
                
                fig = px.bar(cancel_by_cat, x=selected_cat, y='cancel_rate',
                           title=f"Taxa de Cancelamento por {selected_cat}")
                fig.update_layout(yaxis_title="Taxa de Cancelamento (%)")
                st.plotly_chart(fig)
            
            # Tabela de contingência
            st.write("**Tabela de Contingência:**")
            contingency = pd.crosstab(data[selected_cat], data['is_canceled'], margins=True)
            st.dataframe(contingency)
    
    with tab3:
        st.subheader("🔗 Análise de Correlações")
        
        # Matriz de correlação
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig = px.imshow(corr_matrix, 
                       title="Matriz de Correlação",
                       color_continuous_scale="RdBu",
                       aspect="auto")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlações com a variável target
        target_corr = corr_matrix['is_canceled'].abs().sort_values(ascending=False)[1:]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Correlações com Cancelamento (em ordem decrescente):**")
            st.dataframe(target_corr.to_frame('Correlação Absoluta'))
        
        with col2:
            fig = px.bar(x=target_corr.values, y=target_corr.index,
                        orientation='h',
                        title="Correlação com Cancelamento",
                        labels={'x': 'Correlação Absoluta', 'y': 'Variáveis'})
            st.plotly_chart(fig)

elif page == "🤖 Modelagem Preditiva":
    st.header("🤖 Modelagem Preditiva com Regressão Logística")
    
    # Sidebar para configurações do modelo
    st.sidebar.subheader("⚙️ Configurações do Modelo")
    
    test_size = st.sidebar.slider("Tamanho do conjunto de teste", 0.1, 0.5, 0.3, 0.05)
    apply_smote = st.sidebar.checkbox("Aplicar SMOTE", value=True)
    apply_rfe = st.sidebar.checkbox("Aplicar RFE", value=True)
    
    if apply_rfe:
        n_features = st.sidebar.slider("Número de features (RFE)", 5, 20, 12)
    
    random_state = st.sidebar.number_input("Random State", value=42)
    
    # Preparar dados
    df_processed = prepare_data(data)
    
    # Separar features e target
    X = df_processed.drop('is_canceled', axis=1)
    y = df_processed['is_canceled']
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Aplicar SMOTE se selecionado
    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        st.success(f"✅ SMOTE aplicado: {len(X_train):,} → {len(X_train_balanced):,} amostras")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
        st.info("ℹ️ SMOTE não aplicado - dados originais mantidos")
    
    # Aplicar RFE se selecionado
    if apply_rfe:
        rfe = RFE(estimator=LogisticRegression(random_state=random_state), n_features_to_select=n_features)
        X_train_selected = rfe.fit_transform(X_train_balanced, y_train_balanced)
        X_test_selected = rfe.transform(X_test)
        selected_features = X.columns[rfe.support_].tolist()
        
        st.success(f"✅ RFE aplicado: {len(X.columns)} → {len(selected_features)} variáveis")
        
        with st.expander("🔍 Variáveis Selecionadas pelo RFE"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Variáveis Selecionadas:**")
                for feat in selected_features:
                    st.write(f"• {feat}")
            with col2:
                st.write("**Ranking das Variáveis:**")
                ranking_df = pd.DataFrame({
                    'Variável': X.columns,
                    'Ranking': rfe.ranking_,
                    'Selecionada': rfe.support_
                }).sort_values('Ranking')
                st.dataframe(ranking_df.head(10))
    else:
        X_train_selected = X_train_balanced
        X_test_selected = X_test
        selected_features = X.columns.tolist()
    
    # Treinar modelo
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train_selected, y_train_balanced)
    
    # Predições
    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    
    # Métricas de avaliação
    st.subheader("📊 Métricas de Avaliação")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = accuracy_score(y_test, y_pred)
        st.markdown(f"""
        <div class="metric-container">
            <h3>{accuracy:.3f}</h3>
            <p>Acurácia</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        precision = precision_score(y_test, y_pred)
        st.markdown(f"""
        <div class="metric-container">
            <h3>{precision:.3f}</h3>
            <p>Precisão</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        recall = recall_score(y_test, y_pred)
        st.markdown(f"""
        <div class="metric-container">
            <h3>{recall:.3f}</h3>
            <p>Recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        auc = roc_auc_score(y_test, y_pred_proba)
        st.markdown(f"""
        <div class="metric-container">
            <h3>{auc:.3f}</h3>
            <p>AUC-ROC</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráficos de avaliação
    col1, col2 = st.columns(2)
    
    with col1:
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, 
                       text_auto=True, 
                       aspect="auto",
                       title="Matriz de Confusão",
                       labels=dict(x="Predito", y="Real"),
                       x=['Não Cancelado', 'Cancelado'],
                       y=['Não Cancelado', 'Cancelado'])
        st.plotly_chart(fig)
    
    with col2:
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                               name=f'ROC (AUC = {auc:.3f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                               line=dict(dash='dash'), name='Aleatório'))
        fig.update_layout(title="Curva ROC",
                         xaxis_title="Taxa de Falsos Positivos",
                         yaxis_title="Taxa de Verdadeiros Positivos")
        st.plotly_chart(fig)
    
    # Interpretação dos coeficientes
    st.subheader("🔍 Interpretação dos Coeficientes")
    
    if apply_rfe:
        coef_df = pd.DataFrame({
            'Variável': selected_features,
            'Coeficiente': model.coef_[0],
            'Odds Ratio': np.exp(model.coef_[0]),
            'Impacto': ['Aumenta' if x > 0 else 'Diminui' for x in model.coef_[0]]
        })
    else:
        coef_df = pd.DataFrame({
            'Variável': X.columns,
            'Coeficiente': model.coef_[0],
            'Odds Ratio': np.exp(model.coef_[0]),
            'Impacto': ['Aumenta' if x > 0 else 'Diminui' for x in model.coef_[0]]
        })
    
    coef_df['Importância_Abs'] = np.abs(coef_df['Coeficiente'])
    coef_df = coef_df.sort_values('Importância_Abs', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top 10 Variáveis Mais Importantes:**")
        st.dataframe(coef_df.head(10)[['Variável', 'Coeficiente', 'Odds Ratio', 'Impacto']])
    
    with col2:
        # Gráfico de importância
        top_features = coef_df.head(10)
        fig = px.bar(top_features, x='Coeficiente', y='Variável', orientation='h',
                    title="Importância das Variáveis",
                    color='Coeficiente', color_continuous_scale='RdBu')
        st.plotly_chart(fig)
    
    # Análise com Statsmodels
    st.subheader("📈 Análise Estatística Detalhada")
    
    with st.expander("🔍 Ver Resultado Completo do Statsmodels"):
        # Recriar modelo com statsmodels
        if apply_rfe:
            X_sm = pd.DataFrame(X_train_selected, columns=selected_features)
        else:
            X_sm = X_train_balanced
        
        X_sm_const = sm.add_constant(X_sm)
        logit_model = sm.Logit(y_train_balanced, X_sm_const)
        result = logit_model.fit(disp=0)
        
        st.text(str(result.summary()))
        
        # Tabela de coeficientes com significância
        coef_summary = pd.DataFrame({
            'Variável': result.params.index,
            'Coeficiente': result.params.values,
            'Erro Padrão': result.bse.values,
            'z-value': result.tvalues.values,
            'P-valor': result.pvalues.values,
            'Odds Ratio': np.exp(result.params.values),
            'Significante (α=0.05)': result.pvalues.values < 0.05
        })
        
        st.write("**Resumo dos Coeficientes:**")
        st.dataframe(coef_summary)
    
    # Curvas logísticas
    st.subheader("📈 Curvas Logísticas")
    
    # Selecionar variáveis para curvas logísticas
    numeric_selected = [col for col in selected_features if col in data.select_dtypes(include=[np.number]).columns]
    
    if len(numeric_selected) >= 3:
        vars_for_curves = st.multiselect(
            "Selecione até 3 variáveis para gerar curvas logísticas:",
            numeric_selected,
            default=numeric_selected[:3]
        )
        
        if vars_for_curves:
            fig = make_subplots(
                rows=1, cols=len(vars_for_curves),
                subplot_titles=vars_for_curves
            )
            
            for i, var in enumerate(vars_for_curves):
                var_idx = selected_features.index(var) if var in selected_features else None
                if var_idx is not None:
                    var_range = np.linspace(data[var].min(), data[var].max(), 100)
                    
                    # Criar matriz para predição mantendo outras variáveis na média
                    X_pred = np.zeros((100, len(selected_features)))
                    for j, feat in enumerate(selected_features):
                        if feat == var:
                            X_pred[:, j] = var_range
                        else:
                            X_pred[:, j] = X_train_balanced[feat].mean() if feat in X_train_balanced.columns else 0
                    
                    # Predizer probabilidades
                    probs = model.predict_proba(X_pred)[:, 1]
                    
                    fig.add_trace(
                        go.Scatter(x=var_range, y=probs, mode='lines', name=var),
                        row=1, col=i+1
                    )
            
            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(title_text="Valor da Variável")
            fig.update_yaxes(title_text="Probabilidade de Cancelamento")
            st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Validação de Pressupostos":
    st.header("🔍 Validação dos Pressupostos da Regressão Logística")
    
    # Preparar dados para análise
    df_processed = prepare_data(data)
    X = df_processed.drop('is_canceled', axis=1)
    y = df_processed['is_canceled']
    
    st.subheader("1️⃣ Balanceamento da Variável Dependente")
    
    # Análise do balanceamento
    class_counts = y.value_counts()
    balance_ratio = min(class_counts) / max(class_counts)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Classe 0 (Não Cancelado)", class_counts[0])
    with col2:
        st.metric("Classe 1 (Cancelado)", class_counts[1])
    with col3:
        st.metric("Razão de Balanceamento", f"{balance_ratio:.2f}")
    
    if balance_ratio < 0.5:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Aviso:</strong> Os dados estão desbalanceados. 
            Recomenda-se aplicar SMOTE ou outras técnicas de balanceamento.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
            <strong>✅ Bom:</strong> Os dados estão razoavelmente balanceados.
        </div>
        """, unsafe_allow_html=True)
    
    # Visualização do balanceamento
    fig = px.pie(values=class_counts.values, names=['Não Cancelado', 'Cancelado'],
                title="Distribuição da Variável Dependente")
    st.plotly_chart(fig)
    
    st.subheader("2️⃣ Multicolinearidade (VIF)")
    
    # Calcular VIF apenas para variáveis numéricas
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_cols]
    
    if len(X_numeric.columns) > 1:
        vif_df = calculate_vif(X_numeric)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Fatores de Inflação da Variância (VIF):**")
            st.dataframe(vif_df)
            
            # Interpretação do VIF
            high_vif = vif_df[vif_df['VIF'] > 10]
            if len(high_vif) > 0:
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Multicolinearidade detectada:</strong><br>
                    Variáveis com VIF > 10 podem causar problemas de multicolinearidade.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <strong>✅ Sem multicolinearidade:</strong><br>
                    Todas as variáveis têm VIF < 10.
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Gráfico VIF
            fig = px.bar(vif_df, x='VIF', y='Variável', orientation='h',
                        title="Fatores de Inflação da Variância")
            fig.add_vline(x=10, line_dash="dash", line_color="red", 
                         annotation_text="Limite crítico (VIF=10)")
            st.plotly_chart(fig)
    
    st.subheader("3️⃣ Linearidade no Logit")
    
    st.write("""
    Para variáveis contínuas, verificamos se existe uma relação linear entre a variável 
    e o log-odds da variável dependente.
    """)
    
    # Selecionar variáveis contínuas para teste de linearidade
    continuous_vars = st.multiselect(
        "Selecione variáveis contínuas para análise de linearidade:",
        numeric_cols.tolist(),
        default=numeric_cols.tolist()[:3] if len(numeric_cols) >= 3 else numeric_cols.tolist()
    )
    
    if continuous_vars:
        fig = make_subplots(
            rows=1, cols=len(continuous_vars),
            subplot_titles=continuous_vars
        )
        
        for i, var in enumerate(continuous_vars):
            # Criar bins para a variável
            bins = pd.qcut(X[var], q=10, duplicates='drop')
            logit_values = []
            bin_centers = []
            
            for bin_val in bins.cat.categories:
                mask = (X[var] >= bin_val.left) & (X[var] <= bin_val.right)
                if mask.sum() > 0:
                    prob = y[mask].mean()
                    if 0 < prob < 1:  # Evitar log(0) ou log(inf)
                        logit = np.log(prob / (1 - prob))
                        logit_values.append(logit)
                        bin_centers.append((bin_val.left + bin_val.right) / 2)
            
            if len(logit_values) > 2:
                fig.add_trace(
                    go.Scatter(x=bin_centers, y=logit_values, mode='markers+lines', name=var),
                    row=1, col=i+1
                )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title_text="Valor da Variável")
        fig.update_yaxes(title_text="Log-odds")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <strong>💡 Interpretação:</strong> Se os pontos seguem aproximadamente uma linha reta, 
            o pressuposto de linearidade no logit é satisfeito.
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("4️⃣ Outliers e Observações Influentes")
    
    # Análise de outliers usando IQR
    outlier_summary = []
    
    for col in numeric_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
        outlier_pct = (outliers / len(X)) * 100
        
        outlier_summary.append({
            'Variável': col,
            'Outliers': outliers,
            'Percentual': f"{outlier_pct:.1f}%"
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Resumo de Outliers:**")
        st.dataframe(outlier_df)
    
    with col2:
        # Gráfico de outliers
        fig = px.bar(outlier_df, x='Variável', y='Outliers',
                    title="Número de Outliers por Variável")
        st.plotly_chart(fig)
    
    st.subheader("📋 Resumo da Validação de Pressupostos")
    
    # Criar resumo final
    assumptions_check = []
    
    # Balanceamento
    balance_status = "✅ Aprovado" if balance_ratio >= 0.5 else "⚠️ Atenção necessária"
    assumptions_check.append(["Balanceamento", balance_status, f"Razão: {balance_ratio:.2f}"])
    
    # Multicolinearidade
    if len(X_numeric.columns) > 1:
        max_vif = vif_df['VIF'].max()
        vif_status = "✅ Aprovado" if max_vif < 10 else "⚠️ Multicolinearidade detectada"
        assumptions_check.append(["Multicolinearidade", vif_status, f"VIF máximo: {max_vif:.2f}"])
    
    # Outliers
    total_outliers = outlier_df['Outliers'].sum()
    outlier_pct_total = (total_outliers / (len(X) * len(numeric_cols))) * 100
    outlier_status = "✅ Aprovado" if outlier_pct_total < 5 else "⚠️ Muitos outliers"
    assumptions_check.append(["Outliers", outlier_status, f"Total: {total_outliers} ({outlier_pct_total:.1f}%)"])
    
    assumptions_df = pd.DataFrame(assumptions_check, columns=['Pressuposto', 'Status', 'Detalhes'])
    st.dataframe(assumptions_df)

elif page == "💼 Recomendações Estratégicas":
    st.header("💼 Recomendações Estratégicas para Gestão Hoteleira")
    
    # Preparar dados e modelo simplificado para demonstração
    df_processed = prepare_data(data)
    X = df_processed.drop('is_canceled', axis=1)
    y = df_processed['is_canceled']
    
    # Treinar modelo rapidamente
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Calcular feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(model.coef_[0])
    }).sort_values('Importance', ascending=False)
    
    st.subheader("🎯 Fatores de Maior Impacto no Cancelamento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top 10 Fatores Mais Importantes:**")
        top_features = feature_importance.head(10)
        
        for i, row in top_features.iterrows():
            coef = model.coef_[0][X.columns.get_loc(row['Feature'])]
            impact = "Aumenta" if coef > 0 else "Reduz"
            st.write(f"• **{row['Feature']}**: {impact} cancelamento")
    
    with col2:
        fig = px.bar(top_features, x='Importance', y='Feature', orientation='h',
                    title="Importância dos Fatores")
        st.plotly_chart(fig)
    
    st.subheader("🏨 Recomendações por Categoria")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Gestão de Reservas", "💰 Pricing", "👥 Atendimento", "📊 Operacional"])
    
    with tab1:
        st.markdown("""
        <div class="insight-box">
            <h4>📅 Gestão de Reservas</h4>
            
            <strong>Recomendações baseadas no modelo:</strong>
            
            • <strong>Lead Time Alto:</strong> Reservas com lead time > 60 dias têm maior probabilidade de cancelamento
              - Implementar confirmação automática 30 dias antes da chegada
              - Oferecer flexibilidade nas políticas de cancelamento para reservas antecipadas
            
            • <strong>Alterações na Reserva:</strong> Cada mudança aumenta a probabilidade de cancelamento
              - Limitar número de alterações gratuitas
              - Oferecer incentivos para reservas sem alterações
            
            • <strong>Histórico de Cancelamentos:</strong> Clientes com cancelamentos anteriores são de maior risco
              - Solicitar depósito não-reembolsável para clientes com histórico
              - Implementar programa de fidelidade para melhorar retenção
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="insight-box">
            <h4>💰 Estratégias de Pricing</h4>
            
            <strong>Otimização baseada no risco de cancelamento:</strong>
            
            • <strong>Preços Dinâmicos:</strong> Ajustar preços com base no perfil de risco
              - Descontos para perfis de baixo risco de cancelamento
              - Sobretaxa para perfis de alto risco
            
            • <strong>Pacotes de Retenção:</strong> 
              - Oferecer upgrades gratuitos para reservas com baixa probabilidade de cancelamento
              - Criar pacotes "não-reembolsáveis" com desconto significativo
            
            • <strong>Políticas de Cancelamento:</strong>
              - Políticas mais flexíveis para clientes de baixo risco
              - Políticas mais restritivas para segmentos de alto risco
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="insight-box">
            <h4>👥 Estratégias de Atendimento</h4>
            
            <strong>Personalização do atendimento por risco:</strong>
            
            • <strong>Alto Risco de Cancelamento:</strong>
              - Contato proativo 15-7 dias antes da chegada
              - Oferecer serviços adicionais (transfers, restaurantes)
              - Verificar necessidades especiais
            
            • <strong>Solicitações Especiais:</strong> Clientes com solicitações especiais cancelam menos
              - Incentivar solicitações especiais no momento da reserva
              - Facilitar processo de solicitações personalizadas
            
            • <strong>Segmentação de Clientes:</strong>
              - Atendimento VIP para clientes corporativos (menor risco)
              - Acompanhamento especial para grupos (maior risco)
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div class="insight-box">
            <h4>📊 Otimizações Operacionais</h4>
            
            <strong>Gestão de capacidade e overbooking:</strong>
            
            • <strong>Overbooking Inteligente:</strong>
              - Taxa de overbooking baseada no perfil de risco das reservas
              - Monitoramento em tempo real da probabilidade de no-shows
            
            • <strong>Alocação de Quartos:</strong>
              - Priorizar melhores quartos para reservas de baixo risco
              - Manter flexibilidade para upgrades de última hora
            
            • <strong>Staffing:</strong>
              - Ajustar equipe com base na previsão de ocupação real
              - Planejar check-ins com base na probabilidade de chegada
            
            • <strong>Inventory Management:</strong>
              - Liberar quartos com base em previsões de cancelamento
              - Otimizar vendas de última hora
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("📈 Impacto Financeiro Estimado")
    
    # Calcular métricas de impacto
    total_reservas = len(data)
    total_canceladas = data['is_canceled'].sum()
    taxa_cancelamento_atual = total_canceladas / total_reservas
    
    if 'adr' in data.columns:
        adr_medio = data['adr'].mean()
        receita_perdida_atual = total_canceladas * adr_medio
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Taxa de Cancelamento Atual",
                f"{taxa_cancelamento_atual:.1%}",
                help="Percentual de reservas canceladas"
            )
        
        with col2:
            st.metric(
                "Receita Perdida (estimada)",
                f"${receita_perdida_atual:,.0f}",
                help="Receita perdida por cancelamentos"
            )
        
        with col3:
            # Estimativa de melhoria com implementação das recomendações
            melhoria_estimada = 0.15  # 15% de redução nos cancelamentos
            nova_taxa = taxa_cancelamento_atual * (1 - melhoria_estimada)
            receita_recuperada = total_reservas * adr_medio * (taxa_cancelamento_atual - nova_taxa)
            
            st.metric(
                "Receita Recuperável",
                f"${receita_recuperada:,.0f}",
                f"-{melhoria_estimada:.0%} cancelamentos"
            )
    
    st.subheader("🎯 Plano de Implementação")
    
    implementation_plan = pd.DataFrame({
        'Fase': ['Fase 1', 'Fase 2', 'Fase 3', 'Fase 4'],
        'Período': ['0-2 meses', '2-4 meses', '4-8 meses', '8-12 meses'],
        'Ações Principais': [
            'Implementar sistema de scoring de risco, políticas diferenciadas de cancelamento',
            'Desenvolver automação de contatos proativos, sistema de overbooking inteligente',
            'Implementar pricing dinâmico baseado em risco, programa de fidelidade',
            'Otimização contínua com ML, integração com sistemas de gestão'
        ],
        'Investimento': ['Baixo', 'Médio', 'Alto', 'Médio'],
        'ROI Esperado': ['5-10%', '10-15%', '15-25%', '25-35%']
    })
    
    st.dataframe(implementation_plan)

elif page == "🎯 Simulador de Cenários":
    st.header("🎯 Simulador de Cenários")
    
    st.write("""
    Use este simulador para testar diferentes cenários e ver como as mudanças 
    nos parâmetros afetam a probabilidade de cancelamento.
    """)
    
    # Preparar modelo
    df_processed = prepare_data(data)
    X = df_processed.drop('is_canceled', axis=1)
    y = df_processed['is_canceled']
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    st.subheader("🔧 Configure o Cenário")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Características da Reserva:**")
        
        # Inputs principais
        lead_time = st.slider("Lead Time (dias)", 0, 365, 30)
        
        if 'adr' in data.columns:
            adr = st.slider("ADR (Tarifa Média)", 
                           int(data['adr'].min()), 
                           int(data['adr'].max()), 
                           int(data['adr'].mean()))
        else:
            adr = st.slider("ADR (Tarifa Média)", 50, 500, 100)
        
        adults = st.selectbox("Número de Adultos", [1, 2, 3, 4], index=1)
        children = st.selectbox("Número de Crianças", [0, 1, 2, 3])
        
        if 'total_of_special_requests' in data.columns:
            special_requests = st.slider("Solicitações Especiais", 0, 5, 0)
        
        booking_changes = st.slider("Alterações na Reserva", 0, 5, 0)
        
        if 'previous_cancellations' in data.columns:
            prev_cancellations = st.slider("Cancelamentos Anteriores", 0, 5, 0)
    
    with col2:
        st.write("**Características Categóricas:**")
        
        if 'hotel' in data.columns:
            hotel_type = st.selectbox("Tipo de Hotel", data['hotel'].unique())
        
        if 'market_segment' in data.columns:
            market_segment = st.selectbox("Segmento de Mercado", data['market_segment'].unique())
        
        if 'customer_type' in data.columns:
            customer_type = st.selectbox("Tipo de Cliente", data['customer_type'].unique())
        
        if 'arrival_date_month' in data.columns:
            arrival_month = st.selectbox("Mês de Chegada", data['arrival_date_month'].unique())
        
        repeated_guest = st.selectbox("Cliente Repetido", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    
    # Botão para calcular
    if st.button("🔮 Calcular Probabilidade de Cancelamento", type="primary"):
        
        # Criar vetor de características baseado no input
        # Nota: Este é um exemplo simplificado. Na implementação real, 
        # você precisaria criar o vetor completo com todas as features do modelo
        
        # Para demonstração, vamos usar uma aproximação baseada nas variáveis principais
        features_dict = {}
        
        # Preencher com valores padrão e depois atualizar com inputs do usuário
        for col in X.columns:
            if 'lead_time' in col.lower():
                features_dict[col] = lead_time
            elif 'adr' in col.lower():
                features_dict[col] = adr
            elif 'adults' in col.lower():
                features_dict[col] = adults
            elif 'children' in col.lower():
                features_dict[col] = children
            elif 'booking_changes' in col.lower():
                features_dict[col] = booking_changes
            elif 'special_requests' in col.lower() and 'total_of_special_requests' in data.columns:
                features_dict[col] = special_requests
            elif 'previous_cancellations' in col.lower() and 'previous_cancellations' in data.columns:
                features_dict[col] = prev_cancellations
            elif 'repeated_guest' in col.lower():
                features_dict[col] = repeated_guest
            else:
                # Para outras colunas, usar a média ou moda
                if col in data.select_dtypes(include=[np.number]).columns:
                    features_dict[col] = data[col].mean()
                else:
                    features_dict[col] = 0
        
        # Criar array de features
        feature_vector = np.array([features_dict.get(col, 0) for col in X.columns]).reshape(1, -1)
        
        # Fazer predição
        prob_cancelamento = model.predict_proba(feature_vector)[0, 1]
        
        # Exibir resultado
        st.subheader("📊 Resultado da Simulação")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Gauge chart para probabilidade
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_cancelamento * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilidade de Cancelamento (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Classificação de risco
            if prob_cancelamento < 0.25:
                risco = "BAIXO"
                cor = "success"
                icone = "✅"
            elif prob_cancelamento < 0.5:
                risco = "MÉDIO"
                cor = "warning"
                icone = "⚠️"
            elif prob_cancelamento < 0.75:
                risco = "ALTO"
                cor = "warning"
                icone = "🔶"
            else:
                risco = "CRÍTICO"
                cor = "error"
                icone = "🚨"
            
            st.markdown(f"""
            <div class="metric-container">
                <h2>{icone} {risco}</h2>
                <p>Nível de Risco</p>
                <small>Probabilidade: {prob_cancelamento:.1%}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Ações recomendadas
            if prob_cancelamento < 0.25:
                acoes = [
                    "✅ Processar reserva normalmente",
                    "💡 Considerar upgrade gratuito",
                    "📧 Enviar email de confirmação padrão"
                ]
            elif prob_cancelamento < 0.5:
                acoes = [
                    "📞 Contato 48h antes da chegada",
                    "🎁 Oferecer serviços adicionais",
                    "📝 Confirmar detalhes da reserva"
                ]
            elif prob_cancelamento < 0.75:
                acoes = [
                    "🔒 Solicitar garantia adicional",
                    "📞 Contato 72h antes da chegada",
                    "💰 Oferecer upgrade pago"
                ]
            else:
                acoes = [
                    "🚨 Solicitar depósito não-reembolsável",
                    "📞 Contato imediato para confirmação",
                    "🔄 Considerar overbooking para esta vaga"
                ]
            
            st.write("**Ações Recomendadas:**")
            for acao in acoes:
                st.write(f"• {acao}")
        
        # Análise de sensibilidade
        st.subheader("📈 Análise de Sensibilidade")
        
        st.write("Veja como mudanças em variáveis-chave afetam a probabilidade:")
        
        # Testar variação no lead time
        lead_times = np.arange(0, 200, 10)
        probs_lead = []
        
        for lt in lead_times:
            temp_vector = feature_vector.copy()
            # Encontrar índice da coluna lead_time (aproximação)
            lead_time_cols = [i for i, col in enumerate(X.columns) if 'lead_time' in col.lower()]
            if lead_time_cols:
                temp_vector[0, lead_time_cols[0]] = lt
            probs_lead.append(model.predict_proba(temp_vector)[0, 1])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lead_times, y=probs_lead, mode='lines', name='Lead Time'))
        fig.add_hline(y=prob_cancelamento, line_dash="dash", line_color="red", 
                     annotation_text="Cenário Atual")
        fig.update_layout(
            title="Impacto do Lead Time na Probabilidade de Cancelamento",
            xaxis_title="Lead Time (dias)",
            yaxis_title="Probabilidade de Cancelamento"
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🏨 <strong>Sistema de Predição de Cancelamentos</strong><br>
    Desenvolvido para a disciplina de Engenharia de Produção - UnB<br>
    Professor: João Gabriel de Moraes Souza
</div>
""", unsafe_allow_html=True)

# Adicionar informações de como usar no sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 Como usar este dashboard:

1. **🏠 Visão Geral**: Explore os dados gerais
2. **📊 Análise Exploratória**: Analise padrões nos dados
3. **🤖 Modelagem**: Configure e treine modelos
4. **🔍 Validação**: Verifique pressupostos estatísticos
5. **💼 Recomendações**: Veja insights de negócio
6. **🎯 Simulador**: Teste cenários específicos

### 🎯 Principais recursos:
- Análise interativa completa
- Modelagem com SMOTE e RFE
- Validação de pressupostos
- Recomendações estratégicas
- Simulador de cenários
""")

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Dica**: Use o simulador de cenários para testar 
diferentes perfis de clientes e entender como 
otimizar suas estratégias de gestão hoteleira.
""")
