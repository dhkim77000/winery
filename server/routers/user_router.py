from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.orm import Session
from starlette import status
from psycopg2.extensions import connection
from fastapi import FastAPI, Form, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

from crud import create_user
from schema import UserCreate
from database import get_db, get_conn
import pdb
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory='/opt/ml/wine/server/templates')

router = APIRouter(
    prefix="/login",
)

@router.get("/register")
async def get_register_form(request: Request):
    return templates.TemplateResponse('register_form.html', context={'request': request})

@router.post("/register", status_code=status.HTTP_303_SEE_OTHER)
async def user_create(request: Request,
                       email: str = Form(...), 
                       password: str = Form(...), 
                       confirm_password: str = Form(...),
                       db: connection = Depends(get_conn)):

    # Create a new user instance
    user = UserCreate(email=email, password1=password, password2 = confirm_password)
    
    await create_user(db=db, user_create=user)

    return Response(status_code=303, headers={"Location": "/home"})

