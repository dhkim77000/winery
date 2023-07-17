from fastapi import FastAPI, Form, Request, Response
from fastapi import APIRouter , Depends
from psycopg2.extensions import connection
from typing import List
import numpy as np

from database import get_db, get_conn
from crud import get_wine_data,get_wine_data_simple,get_user
from schema import UserAdd
from function import get_top_10_items

#app = FastAPI()


router = APIRouter(
    prefix="/wine"
)


@router.get("/")
async def get_wine_list():
    return {'request': "wine_page"}

# 와인 상세 페이지 정보 
@router.post("/wine_detail")
async def post_wine_info(wine_id_list: List[int],
                         db: connection = Depends(get_conn)):
    wines = {}
    for wine_id in wine_id_list:
        wine = await get_wine_data(db=db, wine_id=wine_id)
        wines[wine_id] = wine
    return wines

# 와인 상세 페이지 정보 
@router.post("/wine_simple")
async def post_wine_info(wine_id_list: List[int],
                         db: connection = Depends(get_conn)):
    wines = {}
    for wine_id in wine_id_list:
        wine = await get_wine_data_simple(db=db, wine_id=wine_id)
        wines[wine_id] = wine
    return wines

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
async def post_wine_simpleinfo(user_wine: UserAdd, db: connection = Depends(get_conn)):
    db_user = await get_user(db=db, email=user_wine.email)
    wine_id_list = db_user.wine_list
    wines = {"인기순": [], "추천순": []}
    popular_wines = get_top_10_items()

    for wine_id in wine_id_list:
        recommend_wine = await get_wine_data_simple(db=db, wine_id=wine_id)
        # 여기서 추천순 기준 wines 딕셔너리에 추가합니다.
        wines["추천순"].append(recommend_wine)

    for wine_id in popular_wines:
        popular_wine =  await get_wine_data_simple(db=db, wine_id=wine_id)
        wines["인기순"].append(popular_wine)

    return wines

