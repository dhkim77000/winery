from fastapi import FastAPI, Form, Request, Response
from fastapi import APIRouter , Depends

from routers import wine_router
from psycopg2.extensions import connection
from typing import List


from database import get_db, get_conn
from crud import get_mbti_data, get_wine_data

#app = FastAPI()


router = APIRouter(
    prefix="/mbti"
)

# mbti 설문지 post 하고
# mbti 결과를 해당 유저의 db에 넣어주어야 함

@router.post("/")
async def post_mbti_question():

    return {'mbti': "mbti_result"}


# 임시 와인 리스트
wine_list = [0,1,2,100,20,30,40,55,77]

@router.get("/{mbti_id}")
async def post_wine_info(mbti_id,
                         db: connection = Depends(get_conn)):
    mbti_data = await get_mbti_data(db=db, mbti_id=mbti_id)
    wine_list = mbti_data.wine_list
    
    wine_info = dict()
    for wine_id in wine_list:
        wine = await get_wine_data(db=db, wine_id=wine_id)
        wine_info['wine_id'] = wine

    return mbti_data ,wine_info 


#router.include_router(wine_router.router)