import os
from typing import List

from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session


import numpy as np
from fastapi import FastAPI, Depends
from schema import PostGet
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd
from catboost import CatBoostClassifier

app = FastAPI()

engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml",
    pool_size = 100, max_overflow = 100)

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

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

catboost_model.load_model(get_model_path('models\catboost_model'))

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    with SessionLocal() as db:
        return db

Base = declarative_base()

class Post(Base):
    __tablename__ = 'post'
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
		id: int,
        time: datetime,
		limit: int = 5,
        db: Session = Depends(get_db)) -> List[PostGet]:
    X = get_df_to_predict(id)

    predicts = pd.DataFrame(catboost_model.predict_proba(X)[:, 1], columns=['like_prob'])
    predicts['post_id'] = posts_data['post_id']
    recommended_post_ids = list(predicts.sort_values(by='like_prob', ascending=False).iloc[:5, 1])

    return db.query(Post)\
                  .filter(Post.id.in_(recommended_post_ids))\
                  .limit(5).all()

