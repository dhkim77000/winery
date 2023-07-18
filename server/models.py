from sqlalchemy import Column, Integer, String, Float, Index, ARRAY, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from database import Base


def create_user_table(db):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS "user" (
        id UUID PRIMARY KEY NOT NULL UNIQUE,
        email VARCHAR NOT NULL UNIQUE,
        password VARCHAR NOT NULL,
        wine_list INTEGER[],
        mbti_result INTEGER[]
    );"""
    print(create_table_query)
    with db.cursor() as cur:
        cur.execute(create_table_query)
        db.commit()


def create_mbiti_table(db):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS "mbti" (
        mbti_id int PRIMARY KEY,
        item text[]
    );
    """
    print(create_table_query)
    with db.cursor() as cur:
        cur.execute(create_table_query)
        db.commit()

def create_wine_table(db_connect):
    create_table_query = """

    CREATE TABLE IF NOT EXISTS "wine" (
        id UUID PRIMARY KEY NOT NULL UNIQUE,
        item_id int,
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
        Gentle int,
        vintage float,
        price float,
        wine_rating float,
        num_votes float,
        country TEXT,
        region TEXT,
        winery TEXT,
        name TEXT,
        wine_style TEXT,
        house TEXT,
        grape TEXT[],
        pairing TEXT[]
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
    wine_list = Column(ARRAY(Integer),  nullable=True)
    mbti_result = Column(ARRAY(Integer),  nullable=True)
    

class MBTI(Base):
    __tablename__ = "mbti"
    mbti_id = Column(Integer, primary_key=True, nullable=False, unique=True)
    wine_list = Column(ARRAY(String),  nullable=False)


class Wine(Base):
    __tablename__ = "wine"
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, unique=True)
    item_id = Column(Integer, primary_key=True, nullable=False, unique=True)
    winetype = Column(String, nullable=True)
    Red_Fruit = Column(Integer, nullable=True)
    Tropical = Column(Integer, nullable=True)
    Tree_Fruit = Column(Integer, nullable=True)
    Oaky = Column(Integer, nullable=True)
    Ageing = Column(Integer, nullable=True)
    Black_Fruit = Column(Integer, nullable=True)
    Citrus = Column(Integer, nullable=True)
    Dried_Fruit = Column(Integer, nullable=True)
    Earthy = Column(Integer, nullable=True)
    Floral = Column(Integer, nullable=True)
    Microbio = Column(Integer, nullable=True)
    Spices = Column(Integer, nullable=True)
    Vegetal = Column(Integer, nullable=True)
    Light = Column(Integer, nullable=True)
    Bold = Column(Integer, nullable=True)
    Smooth = Column(Integer, nullable=True)
    Tannic = Column(Integer, nullable=True)
    Dry = Column(Integer, nullable=True)
    Sweet = Column(Integer, nullable=True)
    Soft = Column(Integer, nullable=True)
    Acidic = Column(Integer, nullable=True)
    Fizzy = Column(Integer, nullable=True)
    Gentle = Column(Integer, nullable=True)
    vintage = Column(Float, nullable=True)
    price = Column(Float, nullable=True)
    wine_rating = Column(Float, nullable=True)
    num_votes = Column(Float, nullable=True)
    country = Column(String, nullable=True)
    region= Column(String, nullable=True)
    winery= Column(String, nullable=True)
    name= Column(String, nullable=True)
    wine_style= Column(String, nullable=True)
    house = Column(String, nullable=True)
    grape = Column(ARRAY(String), nullable=True)
    pairing = Column(ARRAY(String), nullable=True)

    __table_args__ = (
        Index('gin_name_idx', "name", postgresql_ops={'name': 'gin_trgm_ops'}, postgresql_using="gin"),
        Index('gin_house_idx', "house", postgresql_ops={'house': 'gin_trgm_ops'}, postgresql_using="gin"),
    )