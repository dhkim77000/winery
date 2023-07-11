from fastapi import FastAPI
from fastapi import FastAPI, Form, Request
from psycopg2.extras import execute_values, register_uuid

import pandas as pd
import os
import time

from uuid import UUID, uuid4
import models, database, crud

def get_item_data():
    data_path = "/opt/ml/server/winery/data"
    data = pd.read_csv(os.path.join(data_path, "item_df_allfeature.csv"))
    # "grape"
    wine_column = ['winetype','Red Fruit', 'Tropical', 'Tree Fruit', 'Oaky',\
        'Ageing', 'Black Fruit', 'Citrus', 'Dried Fruit', 'Earthy', 'Floral', \
        'Microbio','Spices', 'Vegetal', 'Light', 'Bold', 'Smooth', 'Tannic', 'Dry',\
        'Sweet', 'Soft', 'Acidic', 'Fizzy', 'Gentle']
    
    # EX
    rename_rule = {
        'Red Fruit' : 'Red_Fruit',
        'Tree Fruit': 'Tree_Fruit',
        'Black Fruit':'Black_Fruit',
        'Dried Fruit':'Dried_Fruit',

    }

    
    df = data[wine_column]
    df = data.rename(columns=rename_rule)
    #df['winetype'].fillna(0, inplace=True)
    # for data in range(df.shape[0]):
    #     df['rating'][data] = float(df['rating'][data])
    return df

### CRUD create_wine module 세분화해서 가져오기 
def insert_data(conn, data):
    #pdb.set_trace()
    data['id'] = uuid4() 
    insert_row_query = f"""
    INSERT INTO "wine"
        (id,winetype, Red_Fruit , Tropical, Tree_Fruit, Oaky,
        Ageing, Black_Fruit, Citrus, Dried_Fruit, Earthy, Floral, 
        Microbio,Spices, Vegetal,Light, Bold, Smooth, Tannic,Dry,
        Sweet,Soft,Acidic,Fizzy,Gentle)
        VALUES (
            '{data.id}',
            '{data.winetype}',
            {data.Red_Fruit},
            {data['Tropical']},
            {data['Tree_Fruit']},
            {data['Oaky']},
            {data['Ageing']},
            {data['Black_Fruit']},
            {data['Citrus']},
            {data['Dried_Fruit']},
            {data['Earthy']},
            {data['Floral']},
            {data['Microbio']},
            {data['Spices']},
            {data['Vegetal']},
            {data['Light']},
            {data['Bold']},
            {data['Smooth']},
            {data['Tannic']},
            {data['Dry']},
            {data['Sweet']},
            {data['Soft']},
            {data['Fizzy']},
            {data['Acidic']},
            {data['Gentle']}
        );
    """
    print(insert_row_query)
    with conn.cursor() as cur:
        cur.execute(insert_row_query)
        conn.commit()

def generate_data(conn, df):
    while True:
        register_uuid()
        insert_data(conn, df.sample(1).squeeze())
        time.sleep(1)

if __name__ == "__main__":

    # db 정보 받아오기
    conn = database.get_conn()
    # data table 생성 쿼리 날리기
    models.create_wine_table(conn)
    # data 받아오기
    df = get_item_data()
    # DB에 data넣기
    generate_data(conn, df)

    