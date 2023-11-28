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
from schema import UserCreate, WinePost , Login_User, UserAdd
from models import User, Wine,MBTI, create_user_table,create_wine_table
import pdb , math
from fastapi.security import OAuth2PasswordRequestForm
from psycopg2.extensions import connection
import numpy as np

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



async def get_user(db: connection, email: str,):

    with db.cursor() as cur:
        cur.execute("SELECT * FROM public.user WHERE email = %s", (email,))
        result = cur.fetchone()

    if result is None:
        return False
    else:
        user = Login_User(
            email=result[1],
            password=result[2],
            wine_list = result[3]
        )
        return user
    


    
async def get_user_for_add(new_data:UserAdd, db: connection):
    email = new_data.email
    with db.cursor() as cur:
        cur.execute("SELECT * FROM public.user WHERE email = %s", (email,))
        result = cur.fetchone()

    if result is None:
        raise HTTPException(status_code=404, detail=f"유저 결과가 없습니다.")
    else:
        user = User(
            id=result[0],
            email=result[1],
            password=result[2],
            wine_list= new_data.wine_list
        )
        return mbti

async def search_wine_by_name(db: connection, wine_name):
    min_length = max( int(len(wine_name) * 0.7), 1)

    searched_wine_ids = set()
    wines = []
    with db.cursor() as cur:
        while len(wine_name) >= min_length:
            cur.execute("SELECT item_id, wine_rating, name, region, price, country FROM wine WHERE name ILIKE %s OR house ILIKE %s", ('%' + wine_name + '%', '%' + wine_name + '%'))
            result = cur.fetchall()
            if len(result) != 0: # If result is found, break the loop and return the result
                for wine in result: 
                    if wine[0] not in searched_wine_ids:
                        wine = Wine(
                                item_id= wine[0],
                                wine_rating= wine[1] if not math.isnan(wine[1]) else None,
                                name =  wine[2],
                                region= wine[3],
                                price= wine[4] if not math.isnan(wine[4]) else None,
                                country= wine[5]
                            )
                        
                        searched_wine_ids.add(wine.item_id)
                        wines.append(wine)
                break
            # Reduce the search_term by removing the last character
            wine_name = wine_name[:-1]
    return wines

async def get_wine_datas(db: connection, wine_id):

    recommends = []
    with db.cursor() as cur:
        cur.execute("SELECT * FROM wine WHERE item_id IN %s", (tuple(wine_id),))
        result = cur.fetchall()
        
    if result is None:
        raise HTTPException(status_code=404, detail=f"존재하지 않는 와인입니다.")
    else:
        for wine in result: 
            wine = Wine(
                    item_id= wine[0],
                    wine_rating= wine[1] if not math.isnan(wine[1]) else None,
                    name =  wine[2],
                    region= wine[3],
                    price= wine[4] if not math.isnan(wine[4]) else None,
                    country= wine[5]
                )
            recommends.append(wine)
        
        return recommends

async def get_wine_datas_simple(db: connection, wine_id):

    recommends = []
    with db.cursor() as cur:
        cur.execute("SELECT * FROM wine WHERE item_id IN %s", (tuple(wine_id),))
        result = cur.fetchall()

    if result is None:
        raise HTTPException(status_code=404, detail=f"존재하지 않는 와인입니다.")
    else:
        for wine_info in result: 
            wine = Wine(
                id = wine_info[0],
                item_id = wine_info[1],
                winetype = wine_info[2],
                vintage = wine_info[26] if not math.isnan(wine_info[26]) else None,
                price = wine_info[27] if not math.isnan(wine_info[27]) else None,
                wine_rating = wine_info[28] if not math.isnan(wine_info[28]) else None,
                num_votes = wine_info[-9] if not math.isnan(wine_info[-9]) else None,
                country = wine_info[-8],
                region= wine_info[-7],
                winery= wine_info[-6],
                name= wine_info[-5],
                wine_style= wine_info[-4],
                pairing = wine_info[-1],
            )

            recommends.append(wine)
        

    return recommends
    
    
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


async def check_email_exist(email, cur):

    cur.execute("SELECT * FROM public.user WHERE email = %s", (email,))
    result = cur.fetchone()

    if result is not None:
        return True
    else:
        return False
    

async def create_user(db: connection, user: UserCreate):

    register_uuid()
    db_user = User(id=user.id,
                   password=user.password,
                   email=user.email,
                   wine_list = user.wine_list,
                   mbti_result = user.mbti_result)
    insert_user_query = f"""
    INSERT INTO "{db_user.__tablename__}"
        (id, email, password, wine_list, mbti_result)
        VALUES %s;
        """
    values = [(db_user.id, db_user.email, db_user.password, db_user.wine_list, db_user.mbti_result)]

    
    # Execute the query
    with db.cursor() as cur:
        exist = await check_email_exist(user.email, cur)
        if exist:
            return False
        else:
            print(insert_user_query)
            execute_values(cur, insert_user_query, values)
            db.commit()
            return True
    
#Update
async def set_user_mbti(db: connection, data, topK, min_p, max_p):
    try:
        user = await get_user(db=db, email=data['email'])
        email = user.email

        file_path = '/home/dhkim/server_front/winery_server/server/mbti2idx.json'
        
        with open(file_path, 'r') as file:
            mbti2idx = json.load(file)
 
        mbti = mbti2idx[data['wine_style']]

        # 쿼리 실행
        query = "SELECT item_id FROM wine WHERE item_id IN %s AND price BETWEEN %s AND %s;"
        
        with db.cursor() as cur:
            cur.execute(query, (tuple(topK), min_p, max_p))
            wine_list = cur.fetchall()
            wine_list = [int(id[0]) for id in wine_list]

        update_query = """
        UPDATE "user"
        SET mbti_result = %s,
            wine_list = %s
        WHERE email = %s;
        """
        values = (mbti, wine_list, email)

        with db.cursor() as cur:
            print("Update wine List")
            cur.execute(update_query, values)
            db.commit()
        return True
    
    except Exception as e: 
        print(e)
        return False
    

async def update_wine_list_by_email(db: connection, db_user):
    new_wine_list = db_user.wine_list
    email = db_user.email
    update_query = """
    UPDATE "user"
    SET wine_list = %s
    WHERE email = %s;
    """
    values = (new_wine_list, email)

    # 쿼리 실행
    with db.cursor() as cur:
        cur.execute(update_query, values)
        db.commit()
        print(f"{email}의 wine_list가 업데이트되었습니다.")
    

async def rating_check(collection, email, wine_id):
    if not email:
        raise HTTPException(status_code=400, detail="Missing UID parameter")
    
    # UID가 있는지 확인
    result = collection.find_one({'email': email, 'wine_id': wine_id})
    if result:
        return True
    else:
        return False
    
async def rating_update(collection, email, wine_id, rating, timestamp):
    try:
        data =  {'email': email, 'timestamp': timestamp, 'rating': rating, 'wine_id': wine_id}
        collection.insert_one(data)
        print('Update rating')
        return True
    except Exception as e:
        print(e)
        return False

