from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(
    prefix="/temp",
)


class User(BaseModel):
    email: str
    password: str


class ReturnValue(BaseModel):
    endpoint: str
    status: bool


users = [
    {"email": "seongho@naver.com", "password": "jin"},
    {"email": "yewon@naver.com", "password": "jeon"},
    {"email": "youngseo@naver.com", "password": "kim"},
    {"email": "donghwan@naver.com", "password": "kim"},
    {"email": "jaeseong@naver.com", "password": "park"},
]  # db 대용


@router.get("/db")
def show_user():
    return users


@router.post("/login")
def create_user(user: User):
    retVal = {"endpoint": "/temp/login", "status": False}
    if dict(user) in users:
        retVal["status"] = True
    return retVal


@router.post("/signin", response_model=ReturnValue)
def create_user(user: User):
    retVal = {"endpoint": "/temp/signin", "status": False}
    if not dict(user) in users:
        users.append(user)
        retVal["status"] = True
    return retVal
