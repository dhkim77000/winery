from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.orm import Session
from starlette import status

from database import get_db
from server.crud import create_user
from server.schema import UserCreate
from pydantic import parse_obj_as
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates

import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory='/opt/ml/wine/server/templates')

router = APIRouter(
    prefix="/login",
)

@router.get("/register")
async def get_login_form(request: Request):
    return templates.TemplateResponse('register_form.html', data={'request': request})

@router.post("/register", status_code=status.HTTP_204_NO_CONTENT)
async def user_create(request: Request,
                       email: str = Form(...), 
                       password: str = Form(...), 
                       password2: str = Form(...),
                       db: Session = Depends(get_db)):

    # Create a new user instance
    user = UserCreate(email=email, password1=password, password2 = password2)

    create_user(db=db, user_create=user)

    return templates.TemplateResponse("success.html", {"request": request, "email": email})