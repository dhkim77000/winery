from fastapi import FastAPI, Form, Request, Response
from fastapi import APIRouter , Depends
import json

from routers import wine_router
from psycopg2.extensions import connection
from typing import List

from schema import GetMBTI
from database import get_db, get_conn
from crud import get_mbti_data, get_wine_data
import faiss
import numpy as np
#app = FastAPI()


router = APIRouter(
    prefix="/mbti"
)

def get_avg_vectors(vector_list):
    return np.mean(np.array(vector_list), axis=0)

def faiss_search(to_search, wine_ids, datas):
    if to_search.shape[1] == datas.shape[1]:
        vector_dimension = to_search.shape[1]

        index = faiss.IndexFlatL2(vector_dimension)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(vector_dimension, wine_ids)
        faiss.normalize_L2(datas)
        index.add(datas)

        faiss.normalize_L2(to_search)

        k = index.ntotal
        distances, searched_wine_ids = index.search(to_search, k=k)

        result = []
        for id, d in zip(distances,searched_wine_ids):
            result.append((id, d))
        result.sort(key = lambda x: x[1])

        return result
    
# mbti 설문지 post 하고
# mbti 결과를 해당 유저의 db에 넣어주어야 함

@router.post("/")
async def post_mbti_question():
    return {'mbti': "mbti_result"}

@router.post("/loading")
async def post_mbti_question(mbti_result : GetMBTI):
    answer_list = mbti_result.result
    with open("/opt/ml/wine/server/data/mbti_vectors.json","r") as f:
        answer_vector = json.load(f)

    vector_list = []
    for answer in answer_list:
        vector_list.append(answer_vector[answer])

    mean_vector = await get_avg_vectors(vector_list)

    wine_ids = ##모든 와인에 대한 id
    datas = []  ##모든 와인에 대한 벡터

    search_result = await faiss_search(mean_vector, wine_ids, datas)

    return {'test_result': search_result}


# 임시 와인 리스트
wine_list = [0,1,2,100,20,30,40,55,77]

@router.get("/{mbti_id}")
async def post_wine_info(mbti_id,
                         db: connection = Depends(get_conn)):
    mbti_data = await get_mbti_data(db=db, mbti_id=mbti_id)
    wine_list = mbti_data.wine_list
    
    wine_info = dict()
    for wine_id in wine_list:
        wine = await get_wine_data(db=db, wine_id=wine_id)
        wine_info['wine_id'] = wine

    return mbti_data ,wine_info 


#router.include_router(wine_router.router)