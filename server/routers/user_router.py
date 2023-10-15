from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.orm import Session
from starlette import status
from psycopg2.extensions import connection
from fastapi import FastAPI, Form, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi import HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from datetime import timedelta, datetime
from crud import create_user, get_user, verify_password
from schema import UserCreate
from database import get_db, get_conn
import pdb
import uvicorn

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
SECRET_KEY = "4ab2fce7a6bd79e1c014396315ed322dd6edb1c5d975c6b74a2904135172c03c"
ALGORITHM = "HS256"

app = FastAPI()
templates = Jinja2Templates(directory='//home/dhkim/wineryserver/templates')

router = APIRouter(
    prefix="/login",
)

@router.get("/")
async def get_login_form(request: Request):
    return templates.TemplateResponse('login_form.html', context={'request': request})


@router.post("/", status_code=status.HTTP_303_SEE_OTHER)
async def user_login(request: Request,
                     email: str = Form(...),
                     password: str = Form(...),
                     db: connection = Depends(get_conn)):
    # Check if the user exists in the database
    user = await get_user(db=db, email=email)
    if user and verify_password(password, user.password):
        # User exists and password is correct
        # make access token
        data = {
            "sub": user.id,
            "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        }
        access_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

        response_data =  {
            "access_token": access_token,
            "token_type": "bearer",
            "uid": user.id
            }
        
        return response_data
    else:
        # User does not exist or password is incorrect
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="비밀번호가 일치하지 않거나 존재하지 않는 이메일입니다.",
            headers={"Location": "/login"},
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

    return 



