from fastapi import FastAPI, Form, Request, Response
from fastapi import APIRouter , Depends
from psycopg2.extensions import connection
from database import get_db, get_conn, get_mongo_db
from crud import get_wine_data,get_wine_data_simple,get_user, search_wine_by_name
from crud_mongo import check_rating_datas, rating_update , rating_push, rating_delete
from schema import UserAdd , Usertype, UserInteraction,CheckInteraction , Search
from uuid import UUID, uuid4
from util import get_top_10_items
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

@router.post("/search_by_name")
async def text_search(request: Request, search: Search, page : Optional[int] = None,
                            db: connection = Depends(get_conn)):
    wine = {}
    print(search.name)
    wine_id_lists = await search_wine_by_name(db=db, wine_name = search.name)
    for wine_id in wine_id_lists:         
        wine_contends =  await get_wine_data(db=db, wine_id=wine_id)
        wine[wine_id] = wine_contends
    return wine


# @router.get("/search_by_name")
# async def text_search(wine_name: str = '',
#                             page : Optional[int] = None,
#                             db: connection = Depends(get_conn)):
#     wine_id_lists = await search_wine_by_name(db=db, wine_name = wine_name)
    
#     return wine_id_lists

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

    wine_id_list = db_user.wine_list
    type1,type2 = {},{}
    popular_wines = get_top_10_items()
    if user_wine.type == '인기순':
        for wine_id in popular_wines:
            popular_wine =  await get_wine_data(db=db, wine_id=wine_id)
            type1[wine_id] = popular_wine
        return type1
    
    elif user_wine.type == '추천순':
        for wine_id in wine_id_list:
            recommend_wine = await get_wine_data(db=db, wine_id=wine_id)
            # 여기서 추천순 기준 wines 딕셔너리에 추가합니다.
            type2[wine_id] = recommend_wine
        return type2



@router.post("/rating")
async def update_rating(user_interaction: UserInteraction, 
                        db: Database = Depends(get_mongo_db)):
    
    collection = db.rating
    email = user_interaction.email
    wine_id = user_interaction.wine_id
    rating = user_interaction.rating
    timestamp = user_interaction.timestamp
    

    search_result = check_rating_datas(email,wine_id,db)

    if search_result == "[]":
        push = await rating_push(collection, email, wine_id, rating, timestamp)
        return f"push date complete : {push}"
    else:
        count = await rating_delete(collection, email, wine_id)
        if count == 0 :
            count = False
        push = await rating_push(collection, email, wine_id, rating, timestamp)
        return f"delete {count} and push : {push}"


@router.post("/rating_check")
async def rating_check(check: CheckInteraction, 
                        db: Database = Depends(get_mongo_db)):
    
    #wine_id = user_inter.wine_id
    email = check.email
    wine = check.wine_id
    search_result = await check_rating_datas(email,wine,db)
    #pdb.set_trace()
    
    if search_result == "[]":
        return 0
    else: 
        return search_result
    