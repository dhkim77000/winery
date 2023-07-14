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
import pdb
import uvicorn
from fastapi.responses import RedirectResponse,HTMLResponse
app = FastAPI()


templates = Jinja2Templates(directory='/opt/ml/server/winery/server/templates')

router = APIRouter(
    prefix="/home",
)

# 홈 화면
@router.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    
    return templates.TemplateResponse("success.html", {"request": request})


async def post_mbti_question(request: Request,
                   User1: User 
                   question2: int = Form(...),):
    # mbti 유사도 계산 및 추천 결과 리스트 리턴
    
    return User1.wine_list


@router.get("/{user_id}")
add_user_winelist
async def get_recommend(user_id,
                        db: connection = Depends(get_conn)):
    
    user = await get_user(db=db, user_id=user_id)




