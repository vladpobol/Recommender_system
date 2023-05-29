from typing import List
from schema import Response
from fastapi import FastAPI
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier

from connect_database import engine, Post, Session, SessionLocal
from schema import PostGet

app = FastAPI()

users_data = pd.read_sql('SELECT * from pobol_user10_features',engine).drop('index',axis=1)

posts_df_for_control =  pd.read_sql('SELECT * from pobol_posts_df_proc_control', engine).drop('index', axis=1).rename(columns={'id':'post_id'})
posts_df_for_test = pd.read_sql('SELECT * from pobol_posts_df_proc_test', engine).drop('index', axis=1).rename(columns={'id':'post_id'})

def get_df_to_predict(user_id, exp_group): 
    '''
    Берем данные пользователя, делаем из них N одинаковых строк для каждого поста,
    чтобы получить вероятность для всех постов
    и мерджим с тестовым или контрольным posts_df
    '''

    if exp_group == 'control':
        posts_data = posts_df_for_control
    elif exp_group == 'test':
        posts_data = posts_df_for_test
    else:
        raise ValueError('Unknown group')
    
    users_matrix = np.repeat(users_data[users_data['user_id'] == user_id].values[:1,:],
                             posts_data.shape[0],
                             axis=0)

    X = pd.DataFrame(np.concatenate((users_matrix, posts_data.values), axis=1),
                     columns=users_data.columns.tolist() + posts_data.columns.tolist())
    
    cat_features = ['city', 'exp_group', 'dbscan_clusters', 'kmeans_clusters', 'tf_idf_cluster']

    for col in X.columns:
        if col in cat_features:
            X[col] = X[col].astype(np.int8)
        
    return X.drop(['user_id', 'post_id'], axis=1)


control_catboost = CatBoostClassifier()
test_catboost = CatBoostClassifier()

# заргужаем модели
os.chdir('models')
control_catboost.load_model('control_catboost')
test_catboost.load_model('test_catboost')
os.chdir('..')

# функция для разбиения полтзователей
def md5_hash_partition(user_id):
    # переводим строку в байты
    encoded_id = str(user_id).encode()
    # применяем хэш-функцию
    hashed_id = hashlib.md5(encoded_id)
    # приводим к целочисленоному типу и уменьшаем порядок % 100
    int_hashed_id = int(hashed_id.hexdigest(), 16) % 100
    #разбиваем на котрольную и тестовуювы группы, приблизительно поровну 
    return 'control' if int_hashed_id >= 50 else 'test'


# сам ендпоинт
@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
		id: int,
        time: datetime,
        limit: int = 5) -> List[PostGet]:
    
    exp_group = md5_hash_partition(id)
    # получаем df для предикта
    X = get_df_to_predict(id, exp_group)
    
    if exp_group == 'control':
        model = control_catboost
    elif exp_group == 'test':
        model = test_catboost
    else:
        raise ValueError('Unknown group')
    # делаем предикт нужной моделью
    predicts = pd.DataFrame(model.predict_proba(X)[:, 1], columns=['like_prob'])

    predicts['post_id'] = posts_df_for_control['post_id']
    recommended_post_ids = list(predicts.sort_values(by='like_prob', ascending=False).iloc[:5, 1])
    
     
    with SessionLocal() as db:
        return Response(exp_group=exp_group, 
                         recommendations=db.query(Post)\
                                           .filter(Post.id.in_(recommended_post_ids))\
                                           .limit(limit).all())



