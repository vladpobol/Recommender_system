import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from connect_database import SessionLocal, engine, User_data, Post



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


def posts_processing(input_data):
    '''извлекаем из текста численные признаки с помощью TF-IDF,
       берем топ 100 слова с наибольшей суммой TF-IDF по всем объектам,
       кодируем колонку 'topic' через OneHotEncoder
    '''
    df = input_data.copy()
    tf_idf = TfidfVectorizer()
    tf_idf.fit(df['text'])
    tfidf_dataframe = pd.DataFrame(tf_idf.transform(df['text']).todense(),
                                columns=tf_idf.get_feature_names_out())

    not_important_words = ['that', 'this', 'https', 'with','about','have','what',
                        'than','there','from','will','they','were','which','their',
                        'when','those','these','does']

    important_words = [col for col in tfidf_dataframe.columns\
                    if len(col) > 3 and\
                    (col not in not_important_words)]

    top_100_words = tfidf_dataframe[important_words]\
                                    .sum()\
                                    .sort_values(ascending=False)\
                                    [:100]\
                                    .index.tolist()

    ohe_topic_col = pd.get_dummies(df['topic'], drop_first=True, prefix='topic')

    result_df = pd.concat((tfidf_dataframe[top_100_words], ohe_topic_col), axis=1)
    return result_df.set_index(df['post_id'])\
                    .reset_index()\
                    .rename(columns={'index':'post_id'})


def feed_processing(input_data):
    df = input_data.copy()
    df = df.groupby(['user_id', 'post_id'], as_index=False).sum()
    df['target'] = df['target'].apply(lambda x: int(x > 0))
    df['user_id'].nunique(), df['post_id'].nunique()
    return df


def push_processed_data():
    if __name__ == '__main__':
        with SessionLocal() as session:
            user_data_df = pd.read_sql(session.query(User_data).limit(200000).statement, session.bind)
            posts_df = pd.read_sql(session.query(Post_text).limit(200000).statement, session.bind)

            users_df_processed = users_processing(user_data_df)
            post_df_processed = posts_processing(posts_df)

            users_df_processed.to_sql('pobol_user10_features', con=engine)
            post_df_processed.to_sql('pobol_post10_features', con=engine)

push_processed_data()