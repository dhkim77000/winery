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
from datetime import datetime
import json
from bson.json_util import dumps

from schema import UserCreate, WinePost , Login_User, UserAdd
from models import User, Wine,MBTI, create_user_table,create_wine_table
import pdb , math
from fastapi.security import OAuth2PasswordRequestForm
from psycopg2.extensions import connection
import numpy as np
import pandas as pd
import pymongo ,asyncio

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



async def get_all_rating_data(db):
    # MongoDB에서 rating collection을 가져옵니다.
    collection = db['rating']
    query = {}
    # rating collection에서 모든 데이터를 가져옵니다.
    datas = collection.find(query)
    return datas

async def get_rating_datas(numbers, db):
    collection = db['rating']
    query = {}
    # rating collection에서 모든 데이터를 가져옵니다.
    datas = collection.find(query).head(int(numbers))
    return datas

async def check_rating_datas(user_email,wine,db):
    
    collection = db['rating']
    filter = {'email': user_email,'wine_id':wine}
    # rating collection에서 모든 데이터를 가져옵니다.
    datas = collection.find(filter)
    result_list = list(datas)
    json_result = dumps(result_list)

    return json_result

async def rating_push(collection, email, wine_id, rating, timestamp):
    try:
        # data = [
        # {'uid': 1, 'timestamp': int(datetime.now().timestamp()), 'rating': 4.5, 'wine_id': 1},
        # {'uid': 2, 'timestamp': int(datetime.now().timestamp()), 'rating': 3.8, 'wine_id': 2},
        # {'uid': 1, 'timestamp': int(datetime.now().timestamp()), 'rating': 5.0, 'wine_id': 5}
        # ]
        timestamp = int(datetime.now().timestamp())
        data = {'email': email, 'timestamp': timestamp, 'rating': rating, 'wine_id':  wine_id}
        #pdb.set_trace()
        #collection.insert_many(data)
        collection.insert_one(data)
        return True
    except Exception as e:
        print(e)
        return False
    

async def rating_update(collection, email, wine_id, rating, timestamp):
    try:
        timestamp = int(datetime.now().timestamp())
        update_data = {
        '$set': {
            'timestamp': timestamp,
            'rating': rating
        }
    }
        result = collection.update_many({'email': email, 'wine_id': wine_id}, update_data)
        
        return result
    except Exception as e:
        print(e)
        return False
    


async def rating_delete(collection, email, wine_id):
    try:
        result = await collection.delete_many({'email': email, 'wine_id': wine_id})       
        count = result.deleted_count
        
        return count
    
    except Exception as e:
        print(e)
        return False
    



