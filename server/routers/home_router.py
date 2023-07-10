from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.orm import Session
from starlette import status
from psycopg2.extensions import connection
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates

from crud import create_user
from schema import UserCreate
from database import get_db, get_conn
import pdb
import uvicorn
from fastapi.responses import RedirectResponse,HTMLResponse
app = FastAPI()
templates = Jinja2Templates(directory='/opt/ml/wine/server/templates')

router = APIRouter(
    prefix="/home",
)

# 홈 화면
@router.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    
    return templates.TemplateResponse("success.html", {"request": request})




