from fastapi import FastAPI, Form, Request, Response
from fastapi import APIRouter , Depends
from psycopg2.extensions import connection
from typing import List

from database import get_db, get_conn
from crud import get_wine_data

app = FastAPI()



# 가상의 데이터베이스 리스트
database = [
    {"id": 1, "name": "Wine A" ,"price": 3000},
    {"id": 2, "name": "Wine B","price": 5000},
    {"id": 3, "name": "Wine C","price": 8000}
]

router = APIRouter(
    prefix="/wine"
)


@router.get("/")
async def get_wine_list():
    return {'request': "wine_page"}

@router.get("/{wine_id}")
async def post_wine_info(wine_id,
                         db: connection = Depends(get_conn)):
    wine = await get_wine_data(db=db, wine_id=wine_id)
    

    #database = []
    #for wine in wines:
        #wine_data = {
        #    "id": wine[0],
        #    "name": wine[1],
        #    "price": wine[2]
        #}
        #database.append(wine)

    return wine


@router.get("/{wine_id}")
async def wine_info(wine_id: int):
    wine = next((item for item in database if item["id"] == wine_id), None)
    if wine:
        return {"wine_id": wine_id, "info": wine}
    else:
        return {"error": "Wine not found"}



