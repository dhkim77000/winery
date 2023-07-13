from fastapi import APIRouter
from pydantic import BaseModel
import re

router = APIRouter(
    prefix="/temp",
)

class User(BaseModel):
    email: str
    password: str

users = [] # db 대용

@router.get("/")
def show_user():
    return users

@router.post("/")
def create_user(user: User):
    print(user)
    if re.match(r"\w+@\w+\.[\w,\.]+", user.email):
        users.append(user)
        return {"status": "true"}
    else:
        return {"status": "false"}

