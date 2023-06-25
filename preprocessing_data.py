import pandas as pd
import numpy as np
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from connect_database import engine, get_data_with_sqlalchemy

def users_processing(input_data):
    '''кодирует категориальные фичи,
       те что имеют менее 15 уникальных значений через OneHotEncoding,
       остальные через LabelEncoder
    '''
    df = input_data.copy()

    for col in df.columns:
        
        if df[col].dtype == 'O':
            
            if df[col].nunique() < 15:
                ohe_col = pd.get_dummies(df[col], drop_first=True)
                df = pd.concat((df, ohe_col), axis=1)
                df.drop(col, axis=1, inplace=True)
            else:
                df[col] = LabelEncoder().fit_transform(df[col])
                
    # модель училась на датасете где gender стоит первой колонкой, чтобы не переучивать ее втыкаю костыль
    gender_col = df[['gender']].copy()
    df.drop('gender', axis=1, inplace=True)
    df = pd.concat((gender_col, df), axis=1)
    # конец костыля

    return df

def feed_processing(input_data):
    df = input_data.copy()
    df = df.groupby(['user_id', 'post_id'], as_index=False).sum()
    df['target'] = df['target'].apply(lambda x: int(x > 0))
    df['user_id'].nunique(), df['post_id'].nunique()

    return df

# ПРЕДОБРАБОТКА ТЕКСТА

# Удаляем лишние символы
def del_symbols(text):
    text = text.replace('\n\n', ' ').replace('\n', ' ') # заменяем символы переноса строки на пробел
    text = re.sub(r'[^a-zA-Z\s]', '', text) # убираем все символы кроме букв и пробелов
    text = text.lower() # приводим к нижнему регистру
    
    return text

stop_words = stopwords.words('english')

# Удаляем стоп-слова
def del_stopwords(text):
    important_words = [word for word in text.split() if word not in stop_words]
    
    return ' '.join(important_words)

# Лемматизируем слова
wnl = WordNetLemmatizer()

def lemmatize(text):
    lemm_words = [wnl.lemmatize(word) for word in text.split()]
    
    return ' '.join(lemm_words) 


def get_TFIDF_features(series):
    series = series.apply(del_symbols)\
                   .apply(del_stopwords)\
                   .apply(lemmatize)

    tf_idf = TfidfVectorizer().fit(series)
    # создаем tf-idf индексы для текстов 
    tfidf_dataframe = pd.DataFrame(tf_idf.transform(series).todense(),
                                columns=tf_idf.get_feature_names_out())

    centered = tfidf_dataframe - tfidf_dataframe.mean()
    # Уменьшаем размерность 
    pca_from_tf_idf = PCA(n_components=50).fit_transform(centered)
    
    del tf_idf, centered, tfidf_dataframe
    
    kmeans = KMeans(n_clusters=15, random_state=0).fit(pca_from_tf_idf)

    features_from_tfidf = pd.DataFrame(data=kmeans.transform(pca_from_tf_idf),
                                   columns=[f'DistanceTo{cls}thCluster' for cls in set(kmeans.labels_)])
    features_from_tfidf['tf_idf_cluster'] = kmeans.labels_

    return features_from_tfidf


def get_features_from_embeddings():
    
    os.chdir('BERT_embeddings')
    embeddings = np.load(os.listdir()[-1])
    os.chdir('..')

    centered = embeddings - embeddings.mean()

    # понижаем размерность ембедингов
    pca = PCA(n_components=50)
    pca_decomp = pca.fit_transform(centered)

    # кластеризуем главные компоненты с помощью KMeans
    kmeans = KMeans(n_clusters=15, random_state=0).fit(pca_decomp)

    # кластеризуем ембединги без понижения размерности 
    dbscan = DBSCAN(eps=3).fit(embeddings)

    dists_columns = [f'DistanceToCluster_{i}' for i in range(15)]

    dists_df = pd.DataFrame(
        data=kmeans.transform(pca_decomp),
        columns=dists_columns)

    dists_df['dbscan_clusters'] = dbscan.labels_
    dists_df['kmeans_clusters'] = kmeans.labels_

    return dists_df


def push_processed_data():
    users_df = get_data_with_sqlalchemy('user_data', 200000)
    posts_df = get_data_with_sqlalchemy('posts', 200000)
    
    one_hot_topics = pd.get_dummies(posts_df['topic'], drop_first=True)
    tf_idf_features = get_TFIDF_features(posts_df['text'])
    embeddings_features = get_features_from_embeddings()

    users_df_processed = users_processing(users_df)

    posts_df_control_model = pd.concat((posts_df['id'], 
                                        tf_idf_features,
                                        one_hot_topics), 
                                        axis=1).rename(columns={'id':'post_id'})

    posts_df_test_model = pd.concat((posts_df['id'], # столбец индексов
                                     embeddings_features, # кластеризованные ембединги 
                                     tf_idf_features, # кластеризованные tf-idf
                                     one_hot_topics),
                                     axis=1)
    
    users_df_processed.to_sql('pobol_users_df_proc', con=engine)
    posts_df_control_model.to_sql('pobol_posts_df_proc_control', con=engine)
    posts_df_test_model.to_sql('pobol_posts_df_proc_test', con=engine)
    
if __name__ == '__main__':
    push_processed_data()


