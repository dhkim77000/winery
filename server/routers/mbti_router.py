from fastapi import FastAPI, Form, Request, Response
from fastapi import APIRouter , Depends
import json

from routers import wine_router
from psycopg2.extensions import connection
from typing import List
import pdb
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
    if to_search.shape[0] == datas.shape[1]:
        vector_dimension = to_search.shape[0]

        index = faiss.IndexFlatL2(vector_dimension)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(datas, wine_ids)

        # Faiss expects the query vectors to be normalized
        to_search = np.expand_dims(to_search, axis=0)
        faiss.normalize_L2(to_search)

        k = index.ntotal
        distances, searched_wine_ids = index.search(to_search, k=10)

        result = []
        for ids, dists in zip(searched_wine_ids, distances): 
            result.append(list(zip(ids, dists)))
        
        result.sort(key = lambda x: x[1])
        return result

        
    return None
    
# mbti 설문지 post 하고
# mbti 결과를 해당 유저의 db에 넣어주어야 함

@router.post("/")
async def post_mbti_question():
    return {'mbti': "mbti_result"}

@router.post("/loading")

#######mbti 결과 받아서 미리 계산해둔 벡터에 인덱싱 후-> 평균
#######Wine Vector에 접근해서 FAISS 실행 후 TOP - K 리턴
async def post_mbti_question(mbti_result : GetMBTI):

    #### Example data
    num_wines = 12000
    vector_dimension = 768

    #answer_list = mbti_result.result
    #with open("/opt/ml/wine/server/data/mbti_vectors.json","r") as f:
    #    answer_vector = json.load(f)

    #vector_list = []
    #for answer in answer_list:
    #    vector_list.append(answer_vector[answer])
    vector_list = np.random.rand(num_wines, vector_dimension).astype(np.float32)
    mean_vector = get_avg_vectors(vector_list)
    
    
    wine_ids = np.arange(num_wines)
    datas =  np.random.rand(num_wines, vector_dimension).astype(np.float32)

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