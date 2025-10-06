import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os 
import datetime as dt

# Defina a lista de temas disponíveis (DEVE ser a mesma usada na coleta)
AVAILABLE_THEMES = [
    'Xbox Game Pass', 
    'PlayStation Plus (PSN)', 
    'Nintendo Switch Online'
]

# ----------------------------------------------------
# 1. CONFIGURAÇÃO DA PÁGINA E SELEÇÃO DO TEMA
# ----------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Sentimento Gaming",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Selecione o Tema")

# 1.1. Filtro de Seleção do Tema
selected_theme = st.sidebar.selectbox(
    "Tópico de Análise:",
    options=AVAILABLE_THEMES
)

safe_theme = selected_theme.replace(' ', '_').replace('(', '').replace(')', '')
data_file_name = f'reddit_{safe_theme}_analisado.csv'


# ----------------------------------------------------
# 2. CARREGAMENTO DOS DADOS (CORRIGIDO PARA O FILTRO DE DATA)
# ----------------------------------------------------
@st.cache_data
def load_data(file_path):
    """Carrega o DataFrame, apontando para a pasta Data/."""
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'Data')
    absolute_file_path = os.path.join(data_dir, file_path)
    
    df = pd.read_csv(absolute_file_path, sep=';')
    
    # CORREÇÃO DO FILTRO DE DATA: Normaliza para 00:00:00, mantendo o tipo datetime
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    return df

try:
    df_analise = load_data(data_file_name)
    total_comentarios = len(df_analise)
    
    # ----------------------------------------------------
    # 3. BARRA LATERAL E FILTROS DE DATA
    # ----------------------------------------------------

    data_min = df_analise['date'].min().date()
    data_max = df_analise['date'].max().date()

    date_range = st.sidebar.date_input(
        "Selecione o Intervalo de Tempo:",
        value=(data_min, data_max),
        min_value=data_min,
        max_value=data_max
    )

    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        
        # Filtro com comparação direta entre objetos datetime
        df_filtered = df_analise[
            (df_analise['date'] >= start_date) & 
            (df_analise['date'] <= end_date) 
        ]
    else:
        df_filtered = df_analise


    # ----------------------------------------------------
    # 4. TÍTULO E MÉTRICAS PRINCIPAIS (KPIs)
    # ----------------------------------------------------
    st.title(f"🎮 Análise de Sentimento: {selected_theme}")
    st.subheader("Avaliação do Sentimento em Mídias Sociais (Reddit)")
    st.markdown("---")

    comentarios_filtrados = len(df_filtered)
    distribuicao_sentimento = df_filtered['sentiment'].value_counts(normalize=True).mul(100).round(1)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Total de Comentários na Base de Dados", 
                  value=f"{total_comentarios:,}".replace(",", ".").replace('.', ',').replace(',','.', 1))
    with col2:
        positivo_perc = distribuicao_sentimento.get('Positivo', 0)
        st.metric(label="Percentual Positivo (Filtrado)", value=f"{positivo_perc:.1f}%")

    with col3:
        negativo_perc = distribuicao_sentimento.get('Negativo', 0)
        st.metric(label="Percentual Negativo (Filtrado)", value=f"{negativo_perc:.1f}%")

    st.markdown("---")

    # ----------------------------------------------------
    # 5. GRÁFICO: DISTRIBUIÇÃO DE SENTIMENTO
    # ----------------------------------------------------
    st.header("Distribuição Geral do Sentimento")

    df_distribuicao = distribuicao_sentimento.reset_index()
    df_distribuicao.columns = ['Sentimento', 'Percentual'] 

    fig_pie = px.pie(
        df_distribuicao, 
        names='Sentimento',
        values='Percentual',
        title='Distribuição de Sentimento na Amostra',
        color='Sentimento', 
        color_discrete_map={'Positivo':'#00B894', 
                            'Negativo':'#D63031', 
                            'Neutro':'#6C7A89'},
        hole=.3 
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    # ----------------------------------------------------
    # 6. GRÁFICO: TENDÊNCIA TEMPORAL DE SENTIMENTO (CORRIGIDO)
    # ----------------------------------------------------
    st.header("Tendência Diária do Sentimento")
    
    # CORREÇÃO: Agrupa diretamente pela coluna 'date' (que já é datetime)
    df_tendencia = df_filtered.groupby('date')['sentiment_score'].mean().reset_index()

    fig_line = px.line(
        df_tendencia,
        x='date', # 'date' agora é o tipo correto para time series
        y='sentiment_score',
        title='Sentimento Médio Diário',
        markers=True, 
        line_shape='spline'
    )

    fig_line.add_hline(y=0, line_dash="dash", line_color="gray", 
                       annotation_text="Ponto Neutro", annotation_position="top right")

    fig_line.update_layout(
        xaxis_title="Data",
        yaxis_title="Score de Sentimento Médio",
        yaxis=dict(range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=['Muito Negativo', '-1', 'Neutro', '+1', 'Muito Positivo'])
    )

    st.plotly_chart(fig_line, use_container_width=True)


    # ----------------------------------------------------
    # 7. TABELAS: COMENTÁRIOS MAIS POSITIVOS E NEGATIVOS
    # ----------------------------------------------------
    st.header("Comentários Chave: Positivos vs. Negativos")
    
    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.subheader("🟢 Top 10 Comentários Positivos")
        df_top_pos = df_filtered.sort_values(by='sentiment_score', ascending=False).head(10)
        df_top_pos_display = df_top_pos[['text', 'sentiment_score', 'comment_score']]
        df_top_pos_display.columns = ['Comentário', 'Score de Sentimento', 'Upvotes (Score Reddit)']
        st.dataframe(df_top_pos_display, height=350, use_container_width=True)


    with col_neg:
        st.subheader("🔴 Top 10 Comentários Negativos")
        df_top_neg = df_filtered.sort_values(by='sentiment_score', ascending=True).head(10)
        df_top_neg_display = df_top_neg[['text', 'sentiment_score', 'comment_score']]
        df_top_neg_display.columns = ['Comentário', 'Score de Sentimento', 'Upvotes (Score Reddit)']
        st.dataframe(df_top_neg_display, height=350, use_container_width=True)
        
    st.markdown("---") 
    st.caption(f"Exibindo {comentarios_filtrados} comentários após aplicar os filtros. Base total: {total_comentarios} comentários.")

    # --- FIM DO BLOCO TRY ---

except FileNotFoundError:
    st.error(f"ERRO DE DADOS: O arquivo para o tema '{selected_theme}' não foi encontrado.")
    st.markdown(f"Execute o script de coleta e análise (`coleta_e_analise.py`) para gerar o arquivo **`{data_file_name}`** dentro da pasta **`Data/`**.")
    st.stop()

except Exception as e:
    st.error(f"ERRO DE PROCESSAMENTO: Não foi possível carregar ou processar os dados.")
    st.markdown(f"Detalhes do erro: `{e}`")
    st.stop()