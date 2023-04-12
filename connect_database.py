from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"

# создаём engine
engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_size = 100, max_overflow = 100)
# настройка класса Session c требуемыми настройками
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    with SessionLocal() as db:
        return db

Base = declarative_base()

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


class Post_text(Base):
    __tablename__ = 'post_text_df'
    post_id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)

class Feed_data(Base):
    __tablename__ = 'feed_data'
    timestamp = Column(DateTime, primary_key=True)
    user_id = Column(Integer)
    post_id = Column(Integer)
    action = Column(String)
    target = Column(Integer)

if __name__ == "__main__":
    Base.metadata.create_all(engine)
