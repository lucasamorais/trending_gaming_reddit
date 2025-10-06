import praw
import pandas as pd
import datetime as dt
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

# ----------------------------------------------------
# CONFIGURAÇÃO DE CAMINHO GLOBAL
# ----------------------------------------------------
# O caminho final para os dados é 'Trending Emotionals/Data/'
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Data')

# Garante que a pasta Data exista
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Pasta de dados criada em: {DATA_DIR}")

# ----------------------------------------------------
# 1. FUNÇÕES DE SUPORTE
# ----------------------------------------------------

def download_nltk_resources():
    """Garante que todos os recursos do NLTK estejam instalados."""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('sentiment/vader_lexicon')
        nltk.data.find('tokenizers/punkt_tab')
        print("Recursos do NLTK já instalados.")
    except LookupError:
        print("Baixando recursos necessários do NLTK...")
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('vader_lexicon')
        nltk.download('punkt_tab')

def preprocess_text(text):
    """Limpa o texto para a análise de sentimento."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    tokens = nltk.word_tokenize(text)
    
    stop_words_pt = set(stopwords.words('portuguese'))
    stop_words_en = set(stopwords.words('english'))
    
    filtered_tokens = [word for word in tokens if word not in stop_words_pt and word not in stop_words_en and len(word) > 1]
    
    return ' '.join(filtered_tokens)

def classify_sentiment(score):
    """Classifica o score de sentimento em Positivo, Negativo ou Neutro."""
    if score >= 0.05:
        return 'Positivo'
    elif score <= -0.05:
        return 'Negativo'
    else:
        return 'Neutro'

# ----------------------------------------------------
# 2. FUNÇÃO PRINCIPAL DE COLETA E ANÁLISE POR TEMA
# ----------------------------------------------------

def run_analysis_for_theme(search_term):
    """Executa a coleta, análise de sentimento e salva o arquivo final para um dado termo."""
    
    # 2.1. CONFIGURAÇÃO DE ARQUIVO
    
    safe_term = search_term.replace(' ', '_').replace('(', '').replace(')', '')
    data_file_name = f'reddit_{safe_term}_analisado.csv'
    full_file_path = os.path.join(DATA_DIR, data_file_name) 

    print(f"\n====================== INICIANDO TEMA: {search_term} ======================")

    if os.path.exists(full_file_path):
        print(f"Arquivo '{data_file_name}' já existe. Pulando a coleta e análise.")
        return

    # 2.2. AUTENTICAÇÃO
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    credentials_path = os.path.join(base_dir, 'credentials.txt')

    try:
        with open(credentials_path, 'r') as f:
            creds = [line.strip() for line in f]
            
        CLIENT_ID, CLIENT_SECRET, USERNAME, PASSWORD = creds[:4]
        USER_AGENT = f'Python script for {search_term} analysis by /u/Data-Science-Project'

        reddit = praw.Reddit(
            client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
            username=USERNAME, password=PASSWORD, user_agent=USER_AGENT
        )
        print("Conexão com o Reddit estabelecida.")
    except Exception as e:
        print(f"ERRO DE AUTENTICAÇÃO: Verifique o arquivo credentials.txt em {credentials_path}. Detalhes: {e}")
        return

    # 2.3. COLETA DE DADOS (PRAW)
    posts_data = []
    comments_data = []
    
    SUBREDDITS = 'gaming+xboxone+PS5+NintendoSwitch+playstation' 
    
    print(f"Buscando por '{search_term}' em r/{SUBREDDITS} (100 posts)...")
    
    for submission in reddit.subreddit(SUBREDDITS).search(search_term, limit=100):
        if not submission.stickied:
            # Coleta de Post
            posts_data.append({
                'post_id': submission.id,
                'title': submission.title,
                'score': submission.score,
                'url': submission.url,
                'created_utc': dt.datetime.fromtimestamp(submission.created_utc)
            })
            
            # Coleta de Comentários
            # **********************************************
            # CORREÇÃO CRÍTICA DE VELOCIDADE: limit=0 
            # **********************************************
            submission.comments.replace_more(limit=0) # Limita a expansão para acelerar
            for comment in submission.comments.list():
                if comment.author and comment.body:
                    comments_data.append({
                        'post_id': submission.id,
                        'comment_id': comment.id,
                        'comment_text': comment.body,
                        'comment_score': comment.score,
                        'created_utc': dt.datetime.fromtimestamp(comment.created_utc)
                    })
    
    if not comments_data:
        print(f"AVISO: Nenhuma informação de comentário encontrada para '{search_term}'. Pulando a análise.")
        return

    df_comments = pd.DataFrame(comments_data)
    df_comments.rename(columns={'comment_text': 'text', 'created_utc': 'date'}, inplace=True)
    df = df_comments.dropna(subset=['text'])


    # 2.4. ANÁLISE DE SENTIMENTO
    print(f"Analisando {len(df)} comentários...")
    
    # Pré-processamento
    df['cleaned_text'] = df['text'].astype(str).apply(preprocess_text)

    # VADER
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)

    # 2.5. SALVAMENTO DO ARQUIVO FINAL
    df_final = df[['post_id', 'date', 'text', 'cleaned_text', 'sentiment_score', 'sentiment', 'comment_score']]
    df_final['date'] = pd.to_datetime(df_final['date']).dt.normalize()

    # SALVAMENTO USANDO O CAMINHO E SEPARADOR CORRETO
    df_final.to_csv(full_file_path, index=False, sep=';')

    print(f"Fase 3: Análise Concluída para {search_term}.")
    print(f"Arquivo final salvo em '{full_file_path}'.")

# ----------------------------------------------------
# 3. EXECUÇÃO PRINCIPAL
# ----------------------------------------------------
if __name__ == '__main__':
    
    # 1. Configurar NLTK antes de tudo
    download_nltk_resources()
    
    # 2. Temas escolhidos
    temas_para_analisar = [
        'Xbox Game Pass', 
        'PlayStation Plus (PSN)', 
        'Nintendo Switch Online'
    ]
    
    # 3. Executar o processo para cada tema
    for tema in temas_para_analisar:
        run_analysis_for_theme(tema)

    print("\n===================================================")
    print("TODAS AS ANÁLISES CONCLUÍDAS. TENTE EXECUTAR O DASHBOARD NOVAMENTE!")
    print("===================================================")