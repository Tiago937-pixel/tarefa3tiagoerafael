import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="🏨 Análise de Cancelamento de Reservas Hoteleiras",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                           recall_score, f1_score, confusion_matrix, roc_curve)

# Warnings
import warnings
warnings.filterwarnings('ignore')

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
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🏨 Sistema de Predição de Cancelamentos</h1>
    <h3>Análise com Regressão Logística | Professor João Gabriel</h3>
    <p>Universidade de Brasília - Engenharia de Produção</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Carrega os dados do hotel"""
    try:
        data = pd.read_csv('hotel_bookings.csv')
        if 'is_canceled' in data.columns:
            return data
    except:
        pass
    
    # Upload manual
    uploaded_file = st.sidebar.file_uploader("📁 Upload hotel_bookings.csv", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if 'is_canceled' in data.columns:
            return data
    
    # Dados sintéticos
    st.info("📊 Usando dados sintéticos para demonstração")
    return generate_synthetic_data()

def generate_synthetic_data():
    """Gera dados sintéticos"""
    np.random.seed(42)
    n = 3000  # Reduzido para evitar problemas de memória
    
    # Variáveis principais
    lead_time = np.random.exponential(30, n).astype(int)
    adr = np.random.gamma(2, 50, n)
    adults = np.random.choice([1, 2, 3, 4], n, p=[0.2, 0.6, 0.15, 0.05])
    
    # Criar target baseado em lógica realista
    prob_cancel = (
        0.1 +  # baseline
        0.3 * (lead_time > 60) +  
        0.2 * (adr > 150) +  
        0.1 * (adults == 1)
    )
    prob_cancel = np.clip(prob_cancel, 0, 0.8)
    is_canceled = np.random.binomial(1, prob_cancel, n)
    
    data = pd.DataFrame({
        'hotel': np.random.choice(['Resort Hotel', 'City Hotel'], n, p=[0.3, 0.7]),
        'is_canceled': is_canceled,
        'lead_time': lead_time,
        'arrival_date_month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], n),
        'stays_in_weekend_nights': np.random.poisson(1, n),
        'stays_in_week_nights': np.random.poisson(2, n),
        'adults': adults,
        'children': np.random.choice([0, 1, 2], n, p=[0.8, 0.15, 0.05]),
        'babies': np.random.choice([0, 1], n, p=[0.95, 0.05]),
        'meal': np.random.choice(['BB', 'HB', 'FB'], n, p=[0.7, 0.2, 0.1]),
        'market_segment': np.random.choice(['Online TA', 'Direct', 'Corporate'], n, p=[0.5, 0.3, 0.2]),
        'customer_type': np.random.choice(['Transient', 'Contract'], n, p=[0.8, 0.2]),
        'adr': adr,
        'total_of_special_requests': np.random.poisson(0.5, n),
        'booking_changes': np.random.poisson(0.2, n),
        'previous_cancellations': np.random.poisson(0.1, n),
        'required_car_parking_spaces': np.random.choice([0, 1], n, p=[0.9, 0.1]),
        'is_repeated_guest': np.random.choice([0, 1], n, p=[0.9, 0.1])
    })
    
    return data

def prepare_modeling_data(data):
    """Prepara dados para modelagem de forma robusta"""
    try:
        df = data.copy()
        
        # Selecionar apenas variáveis essenciais para evitar problemas
        essential_numeric = ['lead_time', 'adr', 'adults', 'children', 'babies', 
                           'stays_in_weekend_nights', 'stays_in_week_nights',
                           'total_of_special_requests', 'booking_changes', 
                           'previous_cancellations', 'required_car_parking_spaces', 
                           'is_repeated_guest']
        
        essential_categorical = ['hotel', 'meal', 'market_segment', 'customer_type']
        
        # Filtrar apenas colunas que existem
        numeric_cols = [col for col in essential_numeric if col in df.columns]
        categorical_cols = [col for col in essential_categorical if col in df.columns]
        
        # Começar com variáveis numéricas
        X = df[numeric_cols].copy()
        
        # Adicionar variáveis categóricas de forma controlada
        for col in categorical_cols:
            if col in df.columns:
                # Limitar número de categorias para evitar explosão de dimensionalidade
                top_categories = df[col].value_counts().head(3).index.tolist()
                df_temp = df[col].apply(lambda x: x if x in top_categories else 'Other')
                
                # Criar dummies
                dummies = pd.get_dummies(df_temp, prefix=col, dtype=int)
                X = pd.concat([X, dummies], axis=1)
        
        # Garantir que não há valores missing
        X = X.fillna(0)
        y = df['is_canceled']
        
        return X, y
        
    except Exception as e:
        st.error(f"Erro na preparação dos dados: {e}")
        return None, None

# Carregar dados
data = load_data()

# Sidebar para navegação
st.sidebar.title("📋 Menu")
page = st.sidebar.selectbox(
    "Escolha a análise:",
    ["🏠 Visão Geral", "📊 Análise Exploratória", "🤖 Modelagem e Resultados", "💼 Recomendações"]
)

if page == "🏠 Visão Geral":
    st.header("📈 Visão Geral do Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{len(data):,}</h3>
            <p>Total de Reservas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        cancel_rate = data['is_canceled'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>{cancel_rate * 100:.1f}%</h3>
            <p>Taxa de Cancelamento</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_adr = data['adr'].mean() if 'adr' in data.columns else 0
        st.markdown(f"""
        <div class="metric-container">
            <h3>${avg_adr:.0f}</h3>
            <p>ADR Médio</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_lead_time = data['lead_time'].mean() if 'lead_time' in data.columns else 0
        st.markdown(f"""
        <div class="metric-container">
            <h3>{avg_lead_time:.0f} dias</h3>
            <p>Lead Time Médio</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráficos principais
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Distribuição de Cancelamentos", "Cancelamentos por Hotel"],
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Pizza
    labels = ['Não Cancelada', 'Cancelada']
    values = data['is_canceled'].value_counts()
    fig.add_trace(go.Pie(labels=labels, values=values, hole=0.4), row=1, col=1)
    
    # Barras por hotel
    if 'hotel' in data.columns:
        hotel_cancel = data.groupby('hotel')['is_canceled'].agg(['count', 'mean']).reset_index()
        fig.add_trace(go.Bar(x=hotel_cancel['hotel'], y=hotel_cancel['mean']*100), row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Preview dos dados
    st.subheader("🔍 Preview dos Dados")
    st.dataframe(data.head())

elif page == "📊 Análise Exploratória":
    st.header("📊 Análise Exploratória dos Dados")
    
    tab1, tab2, tab3 = st.tabs(["🔢 Variáveis Numéricas", "📝 Variáveis Categóricas", "🔗 Correlações"])
    
    with tab1:
        st.subheader("Variáveis Numéricas")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_canceled' in numeric_cols:
            numeric_cols.remove('is_canceled')
        
        if len(numeric_cols) > 0:
            selected_numeric = st.multiselect(
                "Selecione variáveis numéricas:",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            )
            
            if selected_numeric:
                # Estatísticas
                st.write("**Estatísticas Descritivas:**")
                st.dataframe(data[selected_numeric].describe())
                
                # Histogramas
                if len(selected_numeric) >= 2:
                    n_vars = min(len(selected_numeric), 4)
                    fig = make_subplots(rows=2, cols=2, subplot_titles=selected_numeric[:n_vars])
                    
                    for i, col in enumerate(selected_numeric[:n_vars]):
                        row = (i // 2) + 1
                        col_pos = (i % 2) + 1
                        fig.add_trace(go.Histogram(x=data[col], name=col), row=row, col=col_pos)
                    
                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Boxplot por cancelamento
                selected_for_box = st.selectbox("Variável para análise detalhada:", selected_numeric)
                fig = px.box(data, x='is_canceled', y=selected_for_box, 
                            title=f"{selected_for_box} por Status de Cancelamento")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Variáveis Categóricas")
        
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            selected_cat = st.selectbox("Selecione uma variável categórica:", categorical_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                value_counts = data[selected_cat].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Distribuição de {selected_cat}")
                st.plotly_chart(fig)
            
            with col2:
                cancel_by_cat = data.groupby(selected_cat)['is_canceled'].mean().reset_index()
                cancel_by_cat['cancel_rate'] = cancel_by_cat['is_canceled'] * 100
                
                fig = px.bar(cancel_by_cat, x=selected_cat, y='cancel_rate',
                           title=f"Taxa de Cancelamento por {selected_cat}")
                st.plotly_chart(fig)
    
    with tab3:
        st.subheader("Matriz de Correlação")
        
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            
            fig = px.imshow(corr_matrix, title="Matriz de Correlação",
                           color_continuous_scale="RdBu", aspect="auto")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            if 'is_canceled' in corr_matrix.columns:
                target_corr = corr_matrix['is_canceled'].abs().sort_values(ascending=False)[1:]
                st.write("**Correlações com Cancelamento:**")
                st.dataframe(target_corr.to_frame('Correlação Absoluta'))

elif page == "🤖 Modelagem e Resultados":
    st.header("🤖 Modelagem com Regressão Logística")
    
    try:
        # Preparar dados
        with st.spinner("Preparando dados..."):
            X, y = prepare_modeling_data(data)
        
        if X is None or y is None:
            st.error("Erro na preparação dos dados")
            st.stop()
        
        st.success(f"Dados preparados: {len(X)} amostras, {len(X.columns)} variáveis")
        
        # Configurações na sidebar
        st.sidebar.subheader("⚙️ Configurações")
        test_size = st.sidebar.slider("Tamanho do teste", 0.1, 0.5, 0.3)
        apply_rfe = st.sidebar.checkbox("Aplicar RFE", value=True)
        
        if apply_rfe:
            max_features = min(15, len(X.columns))
            n_features = st.sidebar.slider("Número de features", 5, max_features, min(10, max_features))
        
        # Divisão dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # RFE se aplicável
        if apply_rfe and len(X.columns) > n_features:
            with st.spinner("Aplicando RFE..."):
                rfe = RFE(LogisticRegression(max_iter=1000, random_state=42), n_features_to_select=n_features)
                X_train_rfe = rfe.fit_transform(X_train, y_train)
                X_test_rfe = rfe.transform(X_test)
                
                selected_features = X.columns[rfe.support_].tolist()
                
                # Converter de volta para DataFrame
                X_train = pd.DataFrame(X_train_rfe, columns=selected_features, index=X_train.index)
                X_test = pd.DataFrame(X_test_rfe, columns=selected_features, index=X_test.index)
        else:
            selected_features = X.columns.tolist()
        
        st.write(f"**Variáveis selecionadas:** {len(selected_features)}")
        with st.expander("Ver variáveis selecionadas"):
            for i, feat in enumerate(selected_features, 1):
                st.write(f"{i}. {feat}")
        
        # Treinar modelo
        with st.spinner("Treinando modelo..."):
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            
            # Predições
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas
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
            auc = roc_auc_score(y_test, y_proba)
            st.markdown(f"""
            <div class="metric-container">
                <h3>{auc:.3f}</h3>
                <p>AUC-ROC</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            # Matriz de confusão
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, title="Matriz de Confusão",
                           x=['Não Cancelado', 'Cancelado'], y=['Não Cancelado', 'Cancelado'])
            st.plotly_chart(fig)
        
        with col2:
            # Curva ROC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {auc:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Aleatório'))
            fig.update_layout(title="Curva ROC", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(fig)
        
        # Coeficientes
        st.subheader("🔍 Interpretação dos Coeficientes")
        
        coef_df = pd.DataFrame({
            'Variável': selected_features,
            'Coeficiente': model.coef_[0],
            'Odds Ratio': np.exp(model.coef_[0]),
            'Impacto': ['Aumenta' if x > 0 else 'Diminui' for x in model.coef_[0]]
        })
        
        coef_df['Importância'] = np.abs(coef_df['Coeficiente'])
        coef_df = coef_df.sort_values('Importância', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 10 Variáveis:**")
            display_df = coef_df.head(10)[['Variável', 'Coeficiente', 'Odds Ratio', 'Impacto']].copy()
            display_df['Coeficiente'] = display_df['Coeficiente'].round(3)
            display_df['Odds Ratio'] = display_df['Odds Ratio'].round(3)
            st.dataframe(display_df)
        
        with col2:
            top_features = coef_df.head(10)
            fig = px.bar(top_features, x='Coeficiente', y='Variável', orientation='h',
                        title="Importância das Variáveis", color='Coeficiente', color_continuous_scale='RdBu')
            st.plotly_chart(fig)
        
        # Curvas logísticas - simplificadas
        st.subheader("📈 Curvas Logísticas")
        
        # Pegar as 3 variáveis numéricas mais importantes
        numeric_features = []
        original_numeric = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for var in coef_df['Variável'].tolist():
            if var in original_numeric and var != 'is_canceled':
                numeric_features.append(var)
            if len(numeric_features) >= 3:
                break
        
        if len(numeric_features) > 0:
            for var in numeric_features:
                if var in data.columns:
                    try:
                        var_min = data[var].min()
                        var_max = data[var].max()
                        var_range = np.linspace(var_min, var_max, 50)
                        
                        # Criar dados para predição
                        X_curve = pd.DataFrame(columns=selected_features)
                        
                        for col in selected_features:
                            if col == var:
                                X_curve[col] = var_range
                            else:
                                X_curve[col] = X_train[col].mean()
                        
                        # Calcular probabilidades
                        probs = model.predict_proba(X_curve)[:, 1]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=var_range, y=probs, mode='lines', name=f'{var}'))
                        fig.update_layout(
                            title=f"Probabilidade de Cancelamento vs {var}",
                            xaxis_title=var,
                            yaxis_title="Probabilidade de Cancelamento",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Não foi possível gerar curva para {var}: {e}")
        else:
            st.info("Nenhuma variável numérica disponível para curvas logísticas")
    
    except Exception as e:
        st.error(f"Erro na modelagem: {e}")
        st.info("Tente ajustar os parâmetros ou verificar os dados")

elif page == "💼 Recomendações":
    st.header("💼 Recomendações Estratégicas")
    
    st.markdown("""
    <div class="insight-box">
        <h4>🎯 Principais Insights</h4>
        
        Com base na análise de regressão logística dos dados de cancelamentos:
        
        <strong>🔍 Fatores de Maior Impacto:</strong>
        • Lead time elevado aumenta risco de cancelamento
        • Histórico de cancelamentos é forte preditor
        • Alterações na reserva indicam incerteza
        • Tipo de cliente influencia decisões
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📅 Gestão", "💰 Pricing", "📊 Operacional"])
    
    with tab1:
        st.markdown("""
        **Gestão de Reservas:**
        
        • **Lead Time Alto:** Implementar confirmação 30 dias antes da chegada
        • **Alterações:** Limitar mudanças gratuitas na reserva
        • **Histórico:** Solicitar depósito para clientes com cancelamentos anteriores
        • **Follow-up:** Contato proativo para reservas de alto risco
        """)
    
    with tab2:
        st.markdown("""
        **Estratégias de Pricing:**
        
        • **Preços Dinâmicos:** Ajustar com base no perfil de risco
        • **Descontos:** Oferecer para perfis de baixo risco
        • **Políticas:** Flexibilizar para clientes confiáveis
        • **Pacotes:** Criar ofertas "não-reembolsáveis" com desconto
        """)
    
    with tab3:
        st.markdown("""
        **Otimizações Operacionais:**
        
        • **Overbooking:** Taxa baseada no perfil de risco das reservas
        • **Alocação:** Priorizar quartos para reservas de baixo risco
        • **Staffing:** Ajustar equipe baseado na ocupação prevista
        • **Inventory:** Liberar quartos com base em previsões
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🏨 <strong>Sistema de Predição de Cancelamentos</strong><br>
    Universidade de Brasília - Engenharia de Produção<br>
    Professor: João Gabriel de Moraes Souza
</div>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 Funcionalidades:

✅ **Visão Geral**: Métricas principais  
✅ **Análise Exploratória**: Filtros interativos  
✅ **Modelagem**: RFE + Métricas + Curvas  
✅ **Recomendações**: Insights estratégicos  

### 🎯 Requisitos Atendidos:
- Regressão Logística ✓
- RFE para seleção ✓  
- Curvas logísticas ✓
- Métricas completas ✓
- Dashboard interativo ✓
""")
