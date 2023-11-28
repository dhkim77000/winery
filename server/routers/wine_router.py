from fastapi import FastAPI, Form, Request, Response
from fastapi import APIRouter , Depends
from psycopg2.extensions import connection
from database import get_db, get_conn, get_mongo_db
from crud import get_wine_data,get_wine_data_simple,get_user, search_wine_by_name, rating_update
from schema import UserAdd, UserInteraction, Usertype
from function import get_top_10_items
from typing import List, Optional
import numpy as np
import pdb
import pymongo
from pymongo.database import Database
#app = FastAPI()


router = APIRouter(
    prefix="/wine"
)


@router.get("/")
async def get_wine_list():
    return {'request': "wine_page"}

# 와인 상세 페이지 정보 
@router.get("/wine_detail{wine_id}")
async def post_wine_info(wine_id: int,
                         db: connection = Depends(get_conn)):
    wines = {}
    wine = await get_wine_data(db=db, wine_id=wine_id)
    return wine

# 와인 상세 페이지 정보 
@router.post("/wine_simple")
async def post_wine_info(wine_id_list: List[int],
                         db: connection = Depends(get_conn)):
    wines = {}
    for wine_id in wine_id_list:
        wine = await get_wine_data_simple(db=db, wine_id=wine_id)
        wines[wine_id] = wine
    return wines

@router.post("/search_by_name")
async def text_search(wine_name: str = '',
                            page : Optional[int] = None,
                            db: connection = Depends(get_conn)):
    result = await search_wine_by_name(db=db, wine_name = wine_name)
    return result

# # 와인 grid 페이지 정보
# @router.post("/wine_recommend")
# async def post_wine_simpleinfo(user_wine : UserAdd,
#                          db: connection = Depends(get_conn)):
#     db_user = await get_user(db = db , email = user_wine.email)
#     wine_id_list = db_user.wine_list
#     wines = {}
#     for wine_id in wine_id_list:
#         wine = await get_wine_data_simple(db=db, wine_id=wine_id)
#         wines[wine_id] = wine
#     return wines

@router.post("/wine_recommend")
async def post_wine_simpleinfo(user_wine: Usertype, db: connection = Depends(get_conn)):
    db_user = await get_user(db=db, email=user_wine.email)
    

    wine_id = db_user.wine_list
    popular_wines = get_top_10_items()
    wine_id.extend(popular_wines)
    wine_id = list(set(wine_id))
    

    wines_json =  await get_wine_data(db=db, wine_id=wine_id)
    print(wines_json)
    return wines_json



@router.post("/rating")
async def update_rating(user_interaction: UserInteraction, 
                        db: Database = Depends(get_mongo_db)):
    
    collection = db.rating

    uid = user_interaction.uid
    wine_id = user_interaction.wine_id
    rating = user_interaction.rating
    timestamp = user_interaction.timestamp
    
    push = await rating_update(collection, uid, wine_id, rating, timestamp)
    return push
