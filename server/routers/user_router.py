from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.orm import Session
from starlette import status
from psycopg2.extensions import connection
from fastapi import FastAPI, Form, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi import HTTPException
from fastapi.security import OAuth2PasswordRequestForm , OAuth2PasswordBearer
from datetime import timedelta, datetime
from crud import create_user, get_user, verify_password , add_mbti_feature
from schema import UserCreate ,Login_User , ReturnValue
from database import get_db, get_conn
from models import User
import pdb,os
import uvicorn

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
SECRET_KEY = "4ab2fce7a6bd79e1c014396315ed322dd6edb1c5d975c6b74a2904135172c03c"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/")
app = FastAPI()
templates = Jinja2Templates(directory=os.getcwd()+'/templates')
router = APIRouter(
    prefix="/login",
)





    # Check if the user exists in the database
    # user = await get_user(db=db, email=user.email)
    # if user:
    #     return True
    # else:
    #     return False
    # if user and verify_password(password, user.password):
    #     # User exists and password is correct
    #     # make access token
    #     data = {
    #         "sub": str(user.id),
    #         "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    #     }
    #     access_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

    #     response_data =  {
    #         "access_token": access_token,
    #         "token_type": "bearer",
    #         "uid": str(user.id)
    #         }
        
    #     return response_data
    # else:
    #     # User does not exist or password is incorrect
    #     return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/register")
async def get_register_form(request: Request):
    return templates.TemplateResponse('register_form.html', context={'request': request})

@router.post("/register", response_model=None)
async def user_create(request: Request, user:UserCreate, db: connection = Depends(get_conn)):

    # Create a new user instance
    retVal = ReturnValue(status=False)
    await create_user(db=db, user_create=user)
    if user:
        retVal.status = True
    else:
        retVal.status = False
    return retVal.status

@router.get("/")
async def get_login_form(request: Request):
    return templates.TemplateResponse('login_form.html', context={'request': request})


@router.post("/login", response_model=None)
async def user_login(request: Request, user : Login_User,
                     db: connection = Depends(get_conn)):
    retVal = ReturnValue(status=False)

    db_user = await get_user(db=db, email=user.email)
    if (db_user.email == user.email) and verify_password(
        user.password, db_user.password
    ):
        retVal.status = True
    else:
        retVal.status = False
    return db_user