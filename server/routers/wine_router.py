from fastapi import FastAPI, Form, Request, Response
from fastapi import APIRouter , Depends
from psycopg2.extensions import connection
from typing import List, Optional
import pdb
from database import get_db, get_conn
from crud import get_wine_data

app = FastAPI()


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
    
    return wine

@router.get("/search_by_name")
async def text_search(wine_name: str = '',
                            page : Optional[int] = None,
                            db: connection = Depends(get_conn)):
    
    wine_id_lists = await search_wine_by_name(db=db, wine_name = wine_name)
    pdb.set_trace()
    return wine_id_lists





