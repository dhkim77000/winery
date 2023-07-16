from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter(
    prefix="/temp",
)


class User(BaseModel):
    email: str
    password: str


class ReturnValue(BaseModel):
    endpoint: str  # frontend 에서 코드상 분리되어있어 필요는 없지만.. 혹시라도 구분해야하는 상황이 생길까?(Login Signin 의 페이지가 합쳐진다거나, ...)
    status: bool


class Rank(BaseModel):
    email: str
    type: str


class ReturnRank(BaseModel):
    ranking: list


users_db = [
    {"email": "seongho@naver.com", "password": "jin"},
    {"email": "yewon@naver.com", "password": "jeon"},
    {"email": "youngseo@naver.com", "password": "kim"},
    {"email": "donghwan@naver.com", "password": "kim"},
    {"email": "jaeseong@naver.com", "password": "park"},
]  # db 대용

ranking_db = {
    "email": "seongho@naver.com",
    "ranks": {
        "type1": [
            {"tier": tier, "wine_name": tier_kor}
            for tier, tier_kor in zip(
                range(1, 11),
                [
                    "일등",
                    "이등",
                    "삼등",
                    "사등",
                    "오등",
                    "육등",
                    "칠등",
                    "팔등",
                    "구등",
                    "십등",
                ],
            )
        ],
        "type2": [
            {"tier": tier, "wine_name": tier_kor}
            for tier, tier_kor in zip(
                range(1, 11),
                [
                    "First",
                    "Second",
                    "Third",
                    "Fourth",
                    "Fifth",
                    "Sixth",
                    "Seventh",
                    "Eighth",
                    "Ninth",
                    "Tenth",
                ],
            )
        ],
    },
}


@router.get("/db")
def show_user():
    return users_db


@router.post("/login", response_model=ReturnValue)
def login_user(user: User):
    retVal = {"endpoint": "/temp/login", "status": False}
    if dict(user) in users_db:
        retVal["status"] = True
    return retVal


@router.post("/signin", response_model=ReturnValue)
def create_user(user: User):
    retVal = {"endpoint": "/temp/signin", "status": False}
    if not dict(user) in users_db:
        users_db.append(user)
        retVal["status"] = True
    return retVal


@router.post("/rank", response_model=ReturnRank)
def post_rank(rank: Rank):
    retVal = {"ranking": []}
    if rank.email == ranking_db["email"]:
        ranks_db = ranking_db["ranks"]
        if rank.type in ranks_db.keys():
            retVal["ranking"].extend(ranks_db[rank.type])
    return retVal
