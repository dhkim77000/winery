from fastapi import FastAPI
from fastapi import APIRouter , Depends 
import pandas as pd
import json 
import pdb, tqdm , csv,ast
from schema import GetMBTI
from datetime import datetime
import faiss
import numpy as np
from psycopg2.extensions import connection
from typing import List, Optional
from routers.wine_router import update_rating
from uuid import UUID, uuid4
from database import get_db, get_conn, get_mongo_db
from pymongo.database import Database


from schema import UserAdd, UserInteraction
from database import get_db, get_conn
from crud import get_user_for_add , update_wine_list_by_email , get_all_wine_feature
from crud_mongo import *
from models import User
import os
pd.set_option('mode.chained_assignment', None)

router = APIRouter(
    prefix="/recommend"
)

def string_data(x):
    str(x)
    return f"{x}" 

def transfrom(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    timestamp = date_obj.timestamp()
    return timestamp


def get_avg_vectors(vector_list):
    return np.mean(np.array(vector_list), axis=0)

def faiss_search(to_search, wine_ids, datas):
    if to_search.shape[0] == datas.shape[1]:
        vector_dimension = to_search.shape[0]

        index = faiss.IndexFlatIP(vector_dimension)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(datas, wine_ids)

        # Faiss expects the query vectors to be normalized
        to_search = np.expand_dims(to_search, axis=0)
        k = index.ntotal
        distances, searched_wine_ids = index.search(to_search, k=10)
        result = []
        for ids, dists in zip(searched_wine_ids, distances): 
            result.append(list(zip(ids, dists)))
        
        result.sort(key = lambda x: x[1])
        return result
    
        
    return None
    
# def sort_wine_by_distance(data):
#     sorted_wine = sorted(data, key=lambda x: x[1], reverse=True)
    
#     return top_10

# /mbti/ test용  
@router.get("/")
async def info():
    return {'page':'mbti_survey'}

def string2array(x):
    x = x.replace('\n', '').strip('[]')
    x_list = [float(i) for i in x.split(' ') if len(i) != 0]
    return np.array(x_list)

@router.post("/loading")

#######mbti 결과 받아서 미리 계산해둔 벡터에 인덱싱 후-> 평균
#######Wine Vector에 접근해서 FAISS 실행 후 TOP - K 리턴
async def post_mbti_question(mbti_result):
    print(mbti_result[2],mbti_result[0])
    item_data = pd.read_csv("/opt/ml/wine/data/item_data (6).csv")

    #winetype 필터링
    if mbti_result[0] == "a1":
        item_data = item_data[(item_data['winetype'] == 'redwine') | (item_data['winetype'] == 'fortifiedwine')]
    elif mbti_result[0] == "a2":
        item_data = item_data[(item_data['winetype'] == 'whitewine') | (item_data['winetype'] == 'roswine')]
    else:
        item_data = item_data[(item_data['winetype'] == 'sparklingwine') | (item_data['winetype'] == 'dessertwine')]

    #price 필터링
    if mbti_result[2] == "c1":
        item_data = item_data[item_data['price'] <= 50]
    elif mbti_result[2] == "c2":
        item_data = item_data[(item_data['price'] > 50) & (item_data['price'] <= 150)]
    elif mbti_result[2] == "c3":
        item_data = item_data[ (item_data['price'] <= 150) & (item_data['price'] <= 500)]
    else:
        item_data = item_data[item_data['price'] > 500]

    #### Example data
    print(item_data.shape)
    num_wines = item_data.shape[0]
    item_data['vectors'] = item_data['vectors'].apply(string2array)

    # print(num_wines)
    answer_list = mbti_result
    with open(os.getcwd()+"/data/mbti_vector.json","r") as f:
        answer_vector = json.load(f)
    vector_list = []
    for answer in list(answer_list):
        vector_list.append(answer_vector[answer])


    # vector_list = np.random.rand(num_wines, vector_dimension).astype(np.float32)
    mean_vector = get_avg_vectors(vector_list)
    
    
    wine_ids = item_data['wine_id'].values

    # 실제 전체 와인들의 벡터
    wine_vectors = []
    for vector in item_data['vectors']: wine_vectors.append(vector)
    wine_vectors = np.array(wine_vectors)

    # wine_ids = np.arange(num_wines) 
    # # 실제 전체 와인들의 벡터
    # datas =  np.random.rand(num_wines, vector_dimension).astype(np.float32)

    # json 스타일로 필터링(T)
    search_result = faiss_search(mean_vector[0], wine_ids, wine_vectors)
    top_10 = [int(x[0]) for x in search_result[0]]

    
    return top_10




#pydantic으로 User, winelist 정보 받아옴

# http://localhost:8000/docs#/default/add_wine_list_to_db_mbti__post 예시
# {
#   "email": "abc123@gmail.com", -> db에 저장되어있는 값이어야 함
#   "wine_list": [0,1,2,100,20,30,40,55,77] -> item_id 값
# }


# mbti 결과를 해당 유저의 db에 넣기
## user 생성시 wine_list값이 None으로 이미 들어가 있으므로 
## 기존 값(None)을 mbti 결과로 update하는 방식

@router.post("/")
async def add_wine_list_to_db(new_data: UserAdd, db: connection = Depends(get_conn)):

    # db에서 기존 데이터를 가져와서(Base) None값인 winelist를 mbti 결과값으로 바꿔줌 
    user = await get_user_for_add(new_data=new_data, db=db)
    # 바꿔준 정보 db에 넣기
    await update_wine_list_by_email(db=db, db_user=user) 


@router.post("/item_to_model")
async def get_wine_to_recbole(db: connection = Depends(get_conn)):
    tuple_data = get_all_wine_feature(db)
    
    wine_column = ['id', 'item_id', 'winetype', 'Red Fruit', 'Tropical', 'Tree Fruit', 'Oaky',
                   'Ageing', 'Black Fruit', 'Citrus', 'Dried Fruit', 'Earthy', 'Floral',
                   'Microbio', 'Spices', 'Vegetal', 'Light', 'Bold', 'Smooth', 'Tannic', 'Dry',
                   'Sweet', 'Soft', 'Acidic', 'Fizzy', 'Gentle', 'vintage', 'price',
                   'wine_rating', 'num_votes', 'country', 'region1', 'grape', 'region2',
                   'region3', 'region4', 'winery', 'name', 'wine_style', 'house', 'pairing']

    result_df = pd.DataFrame(tuple_data, columns=wine_column)
    
    result_df['grape'] = result_df['grape'].apply(string_data)
    result_df['pairing'] = result_df['pairing'].apply(string_data)
    result_df.to_csv("/opt/ml/server/winery/server/data/sample_itemdata_to_recbole.csv",index = False,encoding='utf-8-sig')
    
    return "save file : '/opt/ml/server/winery/server/data/sample_itemdata_to_recbole.csv'"
""

@router.post("/crawl_rating_to_model")
async def preprocess_user_interaction(db: Database = Depends(get_mongo_db)):
    answer = input("csv 불러올건가요 yes or no")
    # 예시로 for 루프를 사용하여 가상의 데이터를 생성하고 처리합니다.
    pdb.set_trace()
    if answer == 'yes':
        print("train_data.inter 파일 불러오기")
        file_path_inter = '/opt/ml/server/winery/server/data/train_data.inter' 
        train_data_inter = pd.read_csv(file_path_inter, sep='\t',encoding='utf-8-sig')
        ####
        #train_data_inter= train_data_inter[:5]
        print("done")
        print("time stamp 값 int형으로 변환")
        train_data_inter['timestamp:float'] = train_data_inter['timestamp:float'].apply(transfrom)
        print("done")
        await rating_data_generator(train_data_inter, db,answer)
        return "save crawl_data"
    else:
        train_data_inter = 0
        await rating_data_generator(train_data_inter, db,answer)

async def rating_data_generator(train_data_inter,db,answer):
    #pdb.set_trace()
    pdb.set_trace()
    # 예시로 for 루프를 사용하여 가상의 데이터를 생성하고 처리합니다.

    for idx in range(train_data_inter.shape[0]):
        result = train_data_inter.iloc[idx,:]
        user_interaction = UserInteraction(
            email = f"user_{result['user_id:token']}@example.com",
            wine_id = result['item_id:token'],
            timestamp = result['timestamp:float'],
            rating = result['user_rating:float']
        )
        # 비동기 함수를 호출합니다.
        push = await update_rating(user_interaction, db)
    

        

    
    numbers = input("'all' or 'number'")
    if numbers == 'all':
        all_rating_data = await get_all_rating_data(db)
        
        inter_column = ["_id","email","timestamp","rating","wine_id"]
        result_df = pd.DataFrame(all_rating_data, columns=inter_column)
        result_df.to_csv("/opt/ml/server/winery/server/data/sample_all_rating_data_to_recbole.csv", index = False, encoding='utf-8-sig')
        return "save all data : '/opt/ml/server/winery/server/data/sample_all_rating_data_to_recbole.csv'"
    
    else:
        rating_datas = await get_rating_datas(numbers,db)
        inter_column = ["_id","email","timestamp","rating","wine_id"]
        result_df = pd.DataFrame(rating_datas, columns=inter_column)
        result_df.to_csv("/opt/ml/server/winery/server/data/sample_rating_data_to_recbole.csv", index = False, encoding='utf-8-sig')
        return "save all data : '/opt/ml/server/winery/server/data/sample_rating_data_to_recbole.csv'"
        






