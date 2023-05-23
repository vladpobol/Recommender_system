from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import psycopg2
import numpy as np
import pandas as pd

DATABASE_URI = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"

engine = create_engine(DATABASE_URI)
# настройка класса Session c требуемыми настройками
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

#создаем классы наших таблиц
class User_data(Base):
    __tablename__ = 'user_data'
    user_id = Column(Integer,primary_key=True)
    age = Column(Integer)
    city = Column(String)
    country = Column(String)
    exp_group = Column(Integer)
    gender = Column(String)
    os = Column(String)
    source = Column(String)


class Post(Base):
    __tablename__ = 'posts'
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)

class Feed_data(Base):
    __tablename__ = 'feed_data'
    timestamp = Column(DateTime, primary_key=True)
    user_id = Column(Integer)
    post_id = Column(Integer)
    action = Column(String)
    target = Column(Integer)

def get_data_with_psycopg(query: str):
    '''с помощью SQL запроса через библиотеку pcycopg2
возвращает резульат запроса в виде numpy.ndarray'''

    conn = psycopg2.connect(DATABASE_URI)
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()

    cursor.close()
    conn.close()

    return np.array(result)

def get_data_with_sqlalchemy(table: str, limit):
    '''С помощью SQLAlchemy и указанием таблица и лимита
возвращает таблицу из базы с заданным лимитом
     
     user == user_data
     post == posts
     feed == feed_data'''

    table_dict = {
            'user': User_data,
            'post': Post,
            'feed': Feed_data
            }
    with SessionLocal() as session:
        data_from_db = session.query(table_dict[table]).limit(limit).all()

    return pd.DataFrame([item.__dict__ for item in data_from_db]).iloc[:,1:]# в первой колонке находятся экземпляры класса нашей таблицы, выкидываем их и оставляем только фичи



if __name__ == "__main__":
    Base.metadata.create_all(engine)

print(type(get_data_with_psycopg('SELECT * from pobol_user10_features')))
