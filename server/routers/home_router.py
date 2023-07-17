from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.orm import Session
from starlette import status
from psycopg2.extensions import connection
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse


from schema import UserCreate
from database import  get_conn
import pdb , os
import uvicorn
from fastapi.responses import  HTMLResponse


from function import get_top_10_items

#app = FastAPI()


templates = Jinja2Templates(directory=os.getcwd()+'/templates')

router = APIRouter(
    prefix="/home",
)

# 홈 화면
@router.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    
    return templates.TemplateResponse("success.html", {"request": request})

# 인기순 api 결과값 노출
@router.get("/popularity", response_class=HTMLResponse)
async def get_popularity(request: Request):
    popularity_result = get_top_10_items()
    
    return JSONResponse({'popularity result': tuple(popularity_result)})







