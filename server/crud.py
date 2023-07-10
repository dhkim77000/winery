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
from schema import UserCreate
from models import User, create_user_table
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
    