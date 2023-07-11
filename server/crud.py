from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from pydantic import BaseModel
from sqlalchemy import func
from passlib.context import CryptContext
from psycopg2.extras import execute_values, register_uuid
from schema import UserCreate, WinePost
#from models import User, Wine,create_user_table, create_wine_table
from models import User,create_user_table,create_wine_table
import pdb
from fastapi.security import OAuth2PasswordRequestForm
from psycopg2.extensions import connection

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create SQLAlchemy engine and session

async def get_user(db: connection, email: str):
    

    with db.cursor() as cur:
        cur.execute("SELECT * FROM public.user WHERE email = %s", (email,))
        result = cur.fetchone()

    if result is None:
        raise HTTPException(status_code=404, detail=f"존재하지 않는 이메일입니다.")
    else:
        user = User(
            id=result[0],
            email=result[1],
            password=result[2],
        )
        return user

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


async def check_email_exist(email, cur):

    cur.execute("SELECT * FROM public.user WHERE email = %s", (email,))
    result = cur.fetchone()

    if result is not None:
        return True
    else:
        return False
    

async def create_user(db: connection, user_create: UserCreate):
    create_user_table(db)
    register_uuid()
    db_user = User(id=user_create.id,
                   password=pwd_context.hash(user_create.password1),
                   email=user_create.email)
    insert_user_query = f"""
    INSERT INTO "{db_user.__tablename__}"
        (id, email, password)
        VALUES %s;
        """
    values = [(db_user.id, db_user.email, db_user.password)]
    
    # Execute the query
    with db.cursor() as cur:
        exist = await check_email_exist(user_create.email, cur)
        if exist:
            raise HTTPException(status_code=404, detail=f"이미 존재하는 이메일입니다.")
        else:
            print(insert_user_query)
            execute_values(cur, insert_user_query, values)
            db.commit()

async def create_wine(db: connection, wine_get: WinePost):
    create_wine_table(db)
    register_uuid()
    db_wine = Wine(id=wine_get.id,
                   winetype=wine_get.winetype,
                   Red_Fruit=wine_get.Red_Fruit,Tropical=wine_get.Tropical,
                   Tree_Fruit=wine_get.Tree_Fruit,Oaky=wine_get.Oaky,
                   Ageing=wine_get.Ageing,Black_Fruit=wine_get.Black_Fruit,
                   Citrus=wine_get.Citrus,Dried_Fruit=wine_get.Dried_Fruit,
                   Earthy=wine_get.Earthy,Floral=wine_get.Floral,
                   Microbio=wine_get.Microbio,Spices=wine_get.Spices,
                   Vegetal=wine_get.Vegetal,Light=wine_get.Light,
                   Bold=wine_get.Bold,Smooth=wine_get.Smooth,
                   Tannic=wine_get.Tannic,Dry=wine_get.Dry,
                   Sweet=wine_get.Sweet,Soft=wine_get.Soft,
                   Gentle=wine_get.Gentle,
                   Acidic=wine_get.Acidic,Fizzy=wine_get.Fizzy)


    insert_wine_query = f"""
    INSERT INTO "{db_wine.__tablename__}"
        (id,winetype,Red_Fruit,Tropical,Tree_Fruit,,Oaky,Ageing,
        Black_Fruit,Citrus,Dried_Fruit,Earthy,Floral,Microbio,Spices,Vegetal,Light,Bold,Smooth,Tannic,Dry,Sweet,Soft,Acidic,Fizzy,Gentle)
        VALUES %s;
        """
    values = [(db_wine.id,
                   db_wine.winetype,
                   db_wine.Red_Fruit,db_wine.Tropical,
                   db_wine.Tree_Fruit,db_wine.Oaky,
                   db_wine.Ageing,db_wine.Black_Fruit,
                   db_wine.Citrus,db_wine.Dried_Fruit,
                   db_wine.Earthy,db_wine.Floral,
                   db_wine.Microbio,db_wine.Spices,
                   db_wine.Vegetal,db_wine.Light,
                   db_wine.Bold,db_wine.Smooth,
                   db_wine.Tannic,db_wine.Dry,
                   db_wine.Sweet,db_wine.Soft,
                   db_wine.Gentle,
                   db_wine.Acidic,db_wine.Fizzy)]
    print(insert_wine_query)
    # Execute the query
    with db.cursor() as cur:
        execute_values(cur, insert_wine_query, values)
        db.commit()
    