import streamlit as st

# Configuração da página DEVE ser a primeira linha
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
from sklearn.preprocessing import LabelEncoder

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
        # Primeiro tenta carregar o arquivo hotel_bookings.csv do repositório
        try:
            data = pd.read_csv('hotel_bookings.csv')
            if 'is_canceled' in data.columns:
                return data
        except Exception as e:
            pass
        
        # Se não conseguir carregar do repositório, permite upload
        uploaded_file = st.sidebar.file_uploader("📁 Upload do arquivo hotel_bookings.csv", type=['csv'])
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            if 'is_canceled' in data.columns:
                return data
        
        # Se não há arquivo, gera dados sintéticos
        st.info("📊 Utilizando dados sintéticos para demonstração.")
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
    children = np.random.choice([0, 1, 2], n, p=[0.8, 0.15, 0.05]).astype(float)
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
    try:
        # Criar cópias para não modificar os dados originais
        df = data.copy()
        
        # Verificar se a coluna target existe
        if 'is_canceled' not in df.columns:
            raise ValueError("Coluna 'is_canceled' não encontrada nos dados")
        
        # Tratar valores missing
        df = df.fillna(0)
        
        # Identificar e processar colunas categóricas
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'is_canceled':
                categorical_cols.append(col)
        
        # Aplicar encoding nas colunas categóricas
        df_processed = df.copy()
        
        for col in categorical_cols:
            if col in df_processed.columns:
                # Usar get_dummies para variáveis categóricas
                dummies = pd.get_dummies(df_processed[col], prefix=col, dtype=int)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                df_processed = df_processed.drop(columns=[col])
        
        # Garantir que todas as colunas (exceto target) sejam numéricas
        for col in df_processed.columns:
            if col != 'is_canceled':
                if df_processed[col].dtype == 'object':
                    # Tentar converter para numérico
                    try:
                        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    except:
                        # Se não conseguir, remover a coluna
                        df_processed = df_processed.drop(columns=[col])
        
        # Remover linhas com valores NaN
        df_processed = df_processed.dropna()
        
        # Verificar se ainda temos dados suficientes
        if len(df_processed) < 100:
            raise ValueError("Dados insuficientes após processamento")
        
        # Verificar se temos variáveis para modelagem
        feature_cols = [col for col in df_processed.columns if col != 'is_canceled']
        if len(feature_cols) < 3:
            raise ValueError("Variáveis insuficientes para modelagem")
        
        return df_processed
        
    except Exception as e:
        raise Exception(f"Erro no processamento: {str(e)}")

# Carregar dados
data = load_data()

# Sidebar para navegação
st.sidebar.title("📋 Navegação")
page = st.sidebar.selectbox(
    "Escolha a análise:",
    ["🏠 Visão Geral", "📊 Análise Exploratória", "🤖 Modelagem Preditiva", 
     "💼 Recomendações Estratégicas", "🎯 Simulador de Cenários"]
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
    
    # Informações sobre as colunas
    st.subheader("📊 Informações das Colunas")
    col_info = pd.DataFrame({
        'Coluna': data.columns,
        'Tipo': [str(dtype) for dtype in data.dtypes],
        'Valores Únicos': [data[col].nunique() for col in data.columns],
        'Valores Faltantes': [data[col].isnull().sum() for col in data.columns],
        '% Faltantes': [f"{data[col].isnull().sum()/len(data)*100:.1f}%" for col in data.columns]
    })
    st.dataframe(col_info)

elif page == "📊 Análise Exploratória":
    st.header("📊 Análise Exploratória dos Dados")
    
    tab1, tab2, tab3 = st.tabs(["🔢 Variáveis Numéricas", "📝 Variáveis Categóricas", "🔗 Correlações"])
    
    with tab1:
        st.subheader("Análise de Variáveis Numéricas")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_canceled' in numeric_cols:
            numeric_cols.remove('is_canceled')
        
        if len(numeric_cols) > 0:
            selected_numeric = st.multiselect(
                "Selecione as variáveis numéricas para análise:",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            )
            
            if selected_numeric:
                # Estatísticas descritivas
                st.write("**Estatísticas Descritivas:**")
                desc_stats = data[selected_numeric].describe()
                st.dataframe(desc_stats)
                
                # Distribuições
                n_vars = min(len(selected_numeric), 4)
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=selected_numeric[:n_vars]
                )
                
                for i, col in enumerate(selected_numeric[:n_vars]):
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
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            
            fig = px.imshow(corr_matrix, 
                           title="Matriz de Correlação",
                           color_continuous_scale="RdBu",
                           aspect="auto")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlações com a variável target
            if 'is_canceled' in corr_matrix.columns:
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
    
    try:
        # Sidebar para configurações do modelo
        st.sidebar.subheader("⚙️ Configurações do Modelo")
        
        test_size = st.sidebar.slider("Tamanho do conjunto de teste", 0.1, 0.5, 0.3, 0.05)
        apply_rfe = st.sidebar.checkbox("Aplicar RFE", value=True)
        
        if apply_rfe:
            n_features = st.sidebar.slider("Número de features (RFE)", 5, 20, 12)
        
        random_state = st.sidebar.number_input("Random State", value=42)
        
        # Preparar dados
        with st.spinner("Preparando dados para modelagem..."):
            df_processed = prepare_data(data)
        
        # Separar features e target
        X = df_processed.drop('is_canceled', axis=1)
        y = df_processed['is_canceled']
        
        # Garantir que todas as features são numéricas
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        st.success(f"Dados preparados: {len(X)} amostras, {len(X.columns)} features")
        
        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Aplicar RFE se selecionado
        selected_features = X.columns.tolist()
        if apply_rfe and len(X.columns) > n_features:
            with st.spinner("Aplicando RFE para seleção de variáveis..."):
                rfe = RFE(estimator=LogisticRegression(random_state=random_state, max_iter=1000), 
                         n_features_to_select=min(n_features, len(X.columns)))
                X_train_selected = rfe.fit_transform(X_train, y_train)
                X_test_selected = rfe.transform(X_test)
                selected_features = X.columns[rfe.support_].tolist()
                
                # Converter arrays de volta para DataFrames
                X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
                X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        else:
            X_train_selected = X_train
            X_test_selected = X_test
        
        with st.expander("🔍 Variáveis Selecionadas"):
            st.write("**Variáveis utilizadas no modelo:**")
            for i, feat in enumerate(selected_features):
                st.write(f"{i+1}. {feat}")
        
        # Treinar modelo
        with st.spinner("Treinando modelo de Regressão Logística..."):
            model = LogisticRegression(random_state=random_state, max_iter=1000)
            model.fit(X_train_selected, y_train)
            
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
        
        coef_df = pd.DataFrame({
            'Variável': selected_features,
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
        
        # Salvar modelo no session state para usar no simulador
        st.session_state['model'] = model
        st.session_state['selected_features'] = selected_features
        st.session_state['coef_df'] = coef_df
        st.session_state['X_train_means'] = X_train_selected.mean().to_dict()
        
        # Curvas logísticas
        st.subheader("📈 Curvas Logísticas")
        
        numeric_selected = [col for col in selected_features if col in data.select_dtypes(include=[np.number]).columns]
        
        if len(numeric_selected) >= 3:
            vars_for_curves = st.multiselect(
                "Selecione até 3 variáveis para gerar curvas logísticas:",
                numeric_selected[:10],
                default=numeric_selected[:3]
            )
            
            if vars_for_curves:
                for var in vars_for_curves:
                    if var in data.columns:
                        # Criar gráfico individual para cada variável
                        var_range = np.linspace(data[var].min(), data[var].max(), 100)
                        
                        # Criar dados fictícios mantendo outras variáveis na média
                        X_curve = pd.DataFrame(columns=selected_features)
                        for col in selected_features:
                            if col == var:
                                X_curve[col] = var_range
                            else:
                                X_curve[col] = X_train_selected[col].mean()
                        
                        # Calcular probabilidades
                        probs = model.predict_proba(X_curve)[:, 1]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=var_range, y=probs, mode='lines', name=f'Curva Logística - {var}'))
                        fig.update_layout(
                            title=f"Probabilidade de Cancelamento vs {var}",
                            xaxis_title=var,
                            yaxis_title="Probabilidade de Cancelamento",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Erro na modelagem: {str(e)}")
        st.info("Verifique se os dados estão no formato correto e tente novamente.")

elif page == "💼 Recomendações Estratégicas":
    st.header("💼 Recomendações Estratégicas para Gestão Hoteleira")
    
    st.markdown("""
    <div class="insight-box">
        <h4>🎯 Principais Insights do Modelo</h4>
        
        Com base na análise dos dados de cancelamentos de reservas hoteleiras, 
        identificamos os seguintes padrões e recomendações estratégicas:
        
        <strong>🔍 Fatores de Maior Impacto:</strong>
        • Lead time elevado (>60 dias) aumenta significativamente o risco de cancelamento
        • Histórico de cancelamentos anteriores é forte preditor
        • Alterações na reserva indicam incerteza do cliente
        • Tipo de cliente e segmento de mercado influenciam decisões
    </div>
    """, unsafe_allow_html=True)
    
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

elif page == "🎯 Simulador de Cenários":
    st.header("🎯 Simulador de Cenários")
    
    st.write("""
    Use este simulador para testar diferentes cenários e ver como as mudanças 
    nos parâmetros afetam a probabilidade de cancelamento.
    """)
    
    # Verificar se o modelo foi treinado
    if 'model' not in st.session_state:
        st.warning("⚠️ Execute primeiro a 'Modelagem Preditiva' para treinar o modelo.")
        st.stop()
    
    try:
        model = st.session_state['model']
        selected_features = st.session_state['selected_features']
        X_train_means = st.session_state['X_train_means']
        
        st.subheader("🔧 Configure o Cenário")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Características da Reserva:**")
            
            # Inputs principais baseados nas colunas disponíveis
            inputs = {}
            
            if 'lead_time' in selected_features:
                inputs['lead_time'] = st.slider("Lead Time (dias)", 0, 365, 30)
            
            if 'adr' in selected_features:
                adr_min = int(data['adr'].min()) if 'adr' in data.columns else 0
                adr_max = int(data['adr'].max()) if 'adr' in data.columns else 500
                adr_mean = int(data['adr'].mean()) if 'adr' in data.columns else 100
                inputs['adr'] = st.slider("ADR (Tarifa Média)", adr_min, adr_max, adr_mean)
            
            if 'adults' in selected_features:
                inputs['adults'] = st.selectbox("Número de Adultos", [1, 2, 3, 4], index=1)
            
            if 'children' in selected_features:
                inputs['children'] = st.selectbox("Número de Crianças", [0, 1, 2, 3])
        
        with col2:
            st.write("**Características Adicionais:**")
            
            if 'total_of_special_requests' in selected_features:
                inputs['total_of_special_requests'] = st.slider("Solicitações Especiais", 0, 5, 0)
            
            if 'booking_changes' in selected_features:
                inputs['booking_changes'] = st.slider("Alterações na Reserva", 0, 5, 0)
            
            if 'previous_cancellations' in selected_features:
                inputs['previous_cancellations'] = st.slider("Cancelamentos Anteriores", 0, 5, 0)
            
            if 'is_repeated_guest' in selected_features:
                inputs['is_repeated_guest'] = st.selectbox("Cliente Repetido", [0, 1], 
                                                        format_func=lambda x: "Não" if x == 0 else "Sim")
        
        # Botão para calcular
        if st.button("🔮 Calcular Probabilidade de Cancelamento", type="primary"):
            
            # Criar vetor de características
            feature_vector = pd.DataFrame(columns=selected_features, index=[0])
            
            for col in selected_features:
                if col in inputs:
                    feature_vector[col] = inputs[col]
                else:
                    # Usar a média do treino para colunas não especificadas
                    feature_vector[col] = X_train_means.get(col, 0)
            
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
                    icone = "✅"
                elif prob_cancelamento < 0.5:
                    risco = "MÉDIO"
                    icone = "⚠️"
                elif prob_cancelamento < 0.75:
                    risco = "ALTO"
                    icone = "🔶"
                else:
                    risco = "CRÍTICO"
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
                    
    except Exception as e:
        st.error(f"Erro no simulador: {str(e)}")
        st.info("Execute novamente a 'Modelagem Preditiva' e tente novamente.")

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
4. **💼 Recomendações**: Veja insights de negócio
5. **🎯 Simulador**: Teste cenários específicos

### 🎯 Principais recursos:
- Análise interativa completa
- Modelagem com RFE
- Recomendações estratégicas
- Simulador de cenários
""")
