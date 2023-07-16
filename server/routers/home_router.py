from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.orm import Session
from starlette import status
from psycopg2.extensions import connection
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates

from crud import create_user , get_user
from schema import UserCreate
from database import get_db, get_conn
import pdb , os
import uvicorn
from fastapi.responses import RedirectResponse, HTMLResponse
from crud import create_user, get_user, verify_password 

app = FastAPI()
templates = Jinja2Templates(directory=os.getcwd()+'/templates')

router = APIRouter(
    prefix="/home",
)

# 홈 화면
@router.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    
    return templates.TemplateResponse("success.html", {"request": request})


# async def post_mbti_question(request: Request,
#                    User1: User 
#                    question2: int = Form(...),):
#     # mbti 유사도 계산 및 추천 결과 리스트 리턴
    
#     return User1.wine_list


# @router.get("/{user_id}")
# add_user_winelist
# async def get_recommend(user_id,
#                         db: connection = Depends(get_conn)):
    
#     user = await get_user(db=db, user_id=user_id)

@router.get("/recommendations")
async def get_register_form(request: Request):
    return templates.TemplateResponse('recommendations_from.html', context={'request': request})

@router.post("/recommendations", status_code=status.HTTP_303_SEE_OTHER)
async def post_item_by_mbti_id(request: Request, email: str = Form(...), mbti_id: int = Form(...), db: connection = Depends(get_conn)):
    # Perform validation on email and mbti_id if needed

    # Call add_mbti_feature function to save item
    result = await add_mbti_feature(email, mbti_id, db)
    return result



