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

app = FastAPI()
templates = Jinja2Templates(directory='/opt/ml/server/winery/server/template')

router = APIRouter(
    prefix="/login",
)

@router.get("/register")
async def get_login_form(request: Request):
    return templates.TemplateResponse('/register_form.html', context={'request': request})

@router.post("/register", status_code=status.HTTP_204_NO_CONTENT)
async def user_create(request: Request,
                       email: str = Form(...), 
                       password: str = Form(...), 
                       password2: str = Form(...),
                       db: connection = Depends(get_conn)):

    # Create a new user instance
    user = UserCreate(email=email, password1=password, password2 = password2)
    
    await create_user(db=db, user_create=user)
    pdb.set_trace()
    return templates.TemplateResponse("/success.html", {"request": request, "email": email})