from fastapi import FastAPI, Form, Request, Response
from fastapi import APIRouter , Depends
from psycopg2.extensions import connection
from typing import List
import numpy as np


from database import get_db, get_conn
from crud import get_wine_data,get_wine_data_simple

app = FastAPI()


router = APIRouter(
    prefix="/wine"
)


@router.get("/")
async def get_wine_list():
    return {'request': "wine_page"}

@router.post("/wine_detail")
async def post_wine_info(wine_id_list: List[int],
                         db: connection = Depends(get_conn)):
    wines = {}
    for wine_id in wine_id_list:
        wine = await get_wine_data(db=db, wine_id=wine_id)

        wines[wine_id] = wine
    return wines


@router.post("/wine_simple")
async def post_wine_simpleinfo(wine_id_list: List[int],
                         db: connection = Depends(get_conn)):
    wines = {}
    for wine_id in wine_id_list:
        wine = await get_wine_data_simple(db=db, wine_id=wine_id)
        wines[wine_id] = wine
    return wines



