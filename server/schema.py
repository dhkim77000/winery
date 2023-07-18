from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from fastapi import HTTPException
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any
from pydantic import EmailStr, validator

from datetime import datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    id: UUID


class UserAdd(BaseModel):
    email: EmailStr
    wine_list : List 

class UserCreate(BaseModel):

    id :UUID = Field(default_factory=uuid4)
    email:EmailStr
    password1: str
    password2: str
    wine_list : Optional[List] = None

    @validator('email', 'password1', 'password2')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise HTTPException(status_code=401, detail="Invalid username or password")
        return v

    @validator('password2')
    def passwords_match(cls, v, values):
        if 'password1' in values and v != values['password1']:
            raise HTTPException(status_code=404, detail=f"비밀번호가 일치하지 않습니다")
       
class UserInteraction(BaseModel):
    uid :UUID
    wine_id : int
    timestamp : int
    rating : float


class WinePost(BaseModel):

    id : UUID = Field(default_factory=uuid4)
    winetype : str
    Red_Fruit : int
    Tropical : int
    Tree_Fruit : int
    Oaky : int
    Ageing : int
    Black_Fruit : int
    Citrus : int
    Dried_Fruit : int
    Earthy : int
    Floral : int
    Microbio : int
    Spices : int
    Vegetal : int
    Light : int
    Bold : int
    Smooth : int
    Tannic : int
    Dry : int
    Sweet : int
    Soft : int
    Acidic : int
    Fizzy : int
    Gentle : int
    vintage : int
    price : int
    wine_rating : int
    num_votes : int

class Login_User(BaseModel):
    id :UUID = Field(default_factory=uuid4)
    email: EmailStr
    password: str
    wine_list : Union[List,None] = None

class ReturnValue(BaseModel):
    status: bool 

class GetMBTI(BaseModel):
    result : List[str]

