
from sqlalchemy import Column, Boolean, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from database import Base


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


class User(Base):
    __tablename__ = "user"

    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)

# class Wine(Base):
#     __tablename__ = "wine"

#     id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, unique=True)
#     winetype = Column(String, nullable=True)
#     Red_Fruit = Column(int, nullable=True)
#     Tropical = Column(int, nullable=True)
#     Tree_Fruit = Column(int, nullable=True)
#     Oaky = Column(int, nullable=True)
#     Ageing = Column(int, nullable=True)
#     Black_Fruit = Column(int, nullable=True)
#     Citrus = Column(int, nullable=True)
#     Dried_Fruit = Column(int, nullable=True)
#     Earthy = Column(int, nullable=True)
#     Floral = Column(int, nullable=True)
#     Microbio = Column(int, nullable=True)
#     Spices = Column(int, nullable=True)
#     Vegetal = Column(int, nullable=True)
#     Light = Column(int, nullable=True)
#     Bold = Column(int, nullable=True)
#     Smooth = Column(int, nullable=True)
#     Tannic = Column(int, nullable=True)
#     Dry = Column(int, nullable=True)
#     Sweet = Column(int, nullable=True)
#     Soft = Column(int, nullable=True)
#     Acidic = Column(int, nullable=True)
#     Fizzy = Column(int, nullable=True)
#     Gentle = Column(int, nullable=True)


