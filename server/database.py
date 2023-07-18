from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import psycopg2
import pymongo

SQLALCHEMY_DATABASE_URL = "postgresql://dhkim:wine123@localhost:5432/server_db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush= True, bind=engine)

Base = declarative_base()

# def get_conn():
#     conn = psycopg2.connect(
#         user="recommy",
#         password="wine123",
#         host="localhost",
#         port=5432,
#         database="winery",
#     )
#     return conn

def get_conn():
    conn = psycopg2.connect(
        user="dhkim",
        password="wine123",
        host="localhost",
        port=5432,
        database="server_db",
    )
    return conn

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_mongo_db():
    mongo_uri = "mongodb://localhost:27017/"
    table_name = 'interaction'
    client = pymongo.MongoClient(mongo_uri)
    db = client[table_name]
    return db  