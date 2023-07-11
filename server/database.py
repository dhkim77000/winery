from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import psycopg2


SQLALCHEMY_DATABASE_URL = "postgresql://dhkim:wine123@localhost:5432/server_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush= True, bind=engine)

Base = declarative_base()

def get_conn():
    conn = psycopg2.connect(
        user="dhkim",
        password="wine123",
        host="localhost",
        port=5432,
        database="server_db",


        # '''
        # CREATE USER dhkim WITH ENCRYPTED PASSWORD 'wine123';
        # GRANT ALL PRIVILEGES ON DATABASE server_db TO dhkim;
        # psql -U dhkim -d server_db
        # '''

    )
    return conn

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

        