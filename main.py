from typing import List
from fastapi import FastAPI, Depends
from datetime import datetime
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

from connect_database import engine, Post, Session, SessionLocal
from schema import PostGet

app = FastAPI()

users_data = pd.read_sql('SELECT * from pobol_user10_features',engine).drop('index',axis=1)
posts_data = pd.read_sql('SELECT * from pobol_post10_features',engine).drop('index',axis=1)

def get_df_to_predict(user_id):
    users_matrix = np.repeat(users_data[users_data['user_id'] == user_id].values[:1,:],
                             7023,
                             axis=0)

    posts_matrix = posts_data.values

    X = pd.DataFrame(np.concatenate((users_matrix, posts_matrix), axis=1),
                     columns=users_data.columns.tolist() + posts_data.columns.tolist())
    X['exp_group'] = X['exp_group'].astype(np.int32)
    X['city'] = X['city'].astype(np.int32)

    return X.drop(['user_id', 'post_id'], axis=1)

catboost_model = CatBoostClassifier()

catboost_model.load_model('models\catboost_model')

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
		id: int) -> List[PostGet]:

    X = get_df_to_predict(id)

    predicts = pd.DataFrame(catboost_model.predict_proba(X)[:, 1], columns=['like_prob'])
    predicts['post_id'] = p osts_data['post_id']
    recommended_post_ids = list(predicts.sort_values(by='like_prob', ascending=False).iloc[:5, 1])

    with SessionLocal() as session:
        return session.query(Post)\
                  .filter(Post.id.in_(recommended_post_ids))\
                  .limit(5).all()

print(recommended_posts(201))