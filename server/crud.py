from fastapi import FastAPI
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
from psycopg2.extensions import connection
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create SQLAlchemy engine and session


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
    print(insert_user_query)
    # Execute the query
    with db.cursor() as cur:
        execute_values(cur, insert_user_query, values)
        db.commit()