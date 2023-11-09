from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.orm import Session
from starlette import status
from psycopg2.extensions import connection
from fastapi import FastAPI, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi import HTTPException
from fastapi.security import OAuth2PasswordRequestForm , OAuth2PasswordBearer
from datetime import timedelta, datetime
from crud import create_user, get_user, verify_password 
from schema import UserCreate ,Login_User , ReturnValue
from database import get_db, get_conn
from models import User
import pdb,os
import uvicorn
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token화 하는 부분 Front에서 하는거면 지워도 되는건가?
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
SECRET_KEY = "4ab2fce7a6bd79e1c014396315ed322dd6edb1c5d975c6b74a2904135172c03c"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/")

#app = FastAPI()

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

# 회원가입
@router.get("/register")
async def get_register_form(request: Request):
    return templates.TemplateResponse('register_form.html', context={'request': request})

# Pydantic UserCrate table로 data 받아서 User class로 db에 넣기
@router.post("/register", response_model=None)
async def user_create(request: Request, user:UserCreate, db: connection = Depends(get_conn)):

    # Create a new user instance
    retVal = ReturnValue(status=False)
    result = await create_user(db=db, user_create=user) #True
    print(result)
    # mbti = await(db=db, user_create=user.email, w)
    ## email, password, mbti_servey -> email,password, mbti_servery,wine_list
    if result:
        retVal.status = True
    else:
        retVal.status = False
    return retVal

# 로그인
@router.get("/")
async def get_login_form(request: Request):
    return templates.TemplateResponse('login_form.html', context={'request': request})


@router.post("/login", response_model=None)
async def user_login(request: Request, user : Login_User,
                     db: connection = Depends(get_conn)):
    retVal = ReturnValue(status=False)
    
    # 유효성 검사 및 해당 유저 정보 유무 확인
    db_user = await get_user(db=db, email=user.email)
    if db_user != 0 :
        if (db_user.email == user.email) and (user.password == db_user.password):
            retVal.status = True
    return retVal