from sqlalchemy import Column, Integer, String, DateTime, create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import psycopg2
import numpy as np
import pandas as pd

DATABASE_URI = # UTI ВАШЕЙ БД

engine = create_engine(DATABASE_URI)
# настройка класса Session c требуемыми настройками
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

inspector = inspect(engine) # чтобы подтянуть названия колонок

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
    
    columns = [desc[0] for desc in cursor.description]
    
    cursor.close()
    conn.close()

    result = pd.DataFrame(data=result,
                        columns=columns)

    if 'index' in columns:
        result.drop('index', axis=1, inplace=True)
    if 'id' in columns:
        result.rename(columns={'id':'post_id'}, inplace=True)

    return result

def get_data_with_sqlalchemy(table_name: str, limit):
    '''С помощью SQLAlchemy и указанием таблица и лимита
возвращает таблицу из базы с заданным лимитом
user_data
posts
feed_data'''

    table_dict = {
            'user_data': User_data,
            'posts': Post,
            'feed_data': Feed_data
            }
    table = table_dict[table_name]

    with SessionLocal() as session:
        data_from_db = session.query(table).limit(limit).all()

    columns_name = [column['name'] for column in inspector.get_columns(table_name)]
    
    return pd.DataFrame([item.__dict__ for item in data_from_db])[columns_name]

if __name__ == '__main__':
  
    Base.metadata.create_all(engine)

