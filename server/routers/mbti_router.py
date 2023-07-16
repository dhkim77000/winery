from fastapi import FastAPI
from fastapi import APIRouter , Depends ,HTTPException, Form, Request, Response

from routers import wine_router
from psycopg2.extensions import connection
from typing import List, Optional
from pydantic import EmailStr, validator


from schema import UserAdd
from database import get_db, get_conn
from crud import get_user_for_add , add_user_info, update_wine_list_by_email
from models import User

#app = FastAPI()


router = APIRouter(
    prefix="/mbti"
)

# mbti 설문지 post 하고
@router.get("/")
async def info():
    return {'page':'mbti_survey'}


# @router.post("/")
# async def post_mbti_question(survey = [1,2,3,4,5]):
#     user:UserCreate

#     return [0,1,2,100,20,30,40,55,77]

# mbti 결과를 해당 유저의 db에 넣어주어야 함

#pydantic으로 User, winelist 정보 받아옴

# 예시
# {
#   "email": "kimyoungseo0330@gmail.com",
#   "wine_list": [0,1,2,100,20,30,40,55,77]
# }

@router.post("/")
async def add_wine_list_to_db(new_data: UserAdd, db: connection = Depends(get_conn)):
    # test 용
    new_data = UserAdd(email="kimyoungseo0330@gmail.com",
               wine_list = [0,1,2,100,20,30,40,55,77] )

    # db에서 기존 데이터를 가져와서(Base) None값인 winelist를 mbti 결과값으로 바꿔줌 
    user = await get_user_for_add(new_data=new_data, db=db)
    # 바꿔준 정보 db에 넣기
    await update_wine_list_by_email(db=db, db_user=user) # 이미 None 값으로 들어가 있는데 넣으면 바뀌어서 들어가나?

