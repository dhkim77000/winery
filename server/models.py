
from sqlalchemy import Column, Boolean, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from database import Base
import csv

def create_user_table(db):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS "user" (
        id UUID PRIMARY KEY NOT NULL UNIQUE,
        email VARCHAR NOT NULL UNIQUE,
        password VARCHAR NOT NULL
    );"""
    print(create_table_query)
    with db.cursor() as cur:
        cur.execute(create_table_query)
        db.commit()

class User(Base):
    __tablename__ = "user"

    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)


def create_wine_table(db_connect):
    create_table_query = """

    CREATE TABLE IF NOT EXISTS "wine" (
        id UUID PRIMARY KEY NOT NULL UNIQUE,
        winetype VARCHAR(20),
        Red_Fruit int ,
        Tropical int,
        Tree_Fruit int,
        Oaky int,
        Ageing int,
        Black_Fruit int,
        Citrus int,
        Dried_Fruit int,
        Earthy int,
        Floral int,
        Microbio int,
        Spices int,
        Vegetal int,
        Light int,
        Bold int,
        Smooth int,
        Tannic int,
        Dry int,
        Sweet int ,
        Soft int,
        Acidic int,
        Fizzy int,
        Gentle int
    );
    """
    print(create_table_query)
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()




