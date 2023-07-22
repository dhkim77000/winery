from fastapi import FastAPI
from fastapi import FastAPI, Form, Request
from psycopg2.extras import execute_values, register_uuid
from psycopg2.extensions import connection
from tqdm import tqdm


import pandas as pd
import os
import time
import csv
import sys
import pdb
import argparse
import copy

from uuid import UUID, uuid4
import models, database, crud

pd.set_option('mode.chained_assignment', None)
# user
# ['user_id:token', 'count:float', 'mean:float']
# ['0', '46', '4.145652173913043']
# ['1', '105', '4.124761904761905']
# ['2', '138', '4.023188405797102']
# ['3', '942', '4.00276008492569']
# inter
# ['user_id:token', 'user_rating:float', 'timestamp:float', 'item_id:token']
# ['0', '4.1', '2022-07-25 00:00:00', '65527']
# ['1', '4.4', '2022-07-24 00:00:00', '65527']
# ['2', '4.4', '2022-09-23 00:00:00', '65527']
# ['3', '4.0', '2022-05-16 00:00:00', '65527']


def delete_table(cursor, table_name):
    
    # 기존 테이블 삭제 (선택사항)
    #cursor = conn.cursor()
    cursor.execute(f'DROP TABLE IF EXISTS "{table_name}";')
    cursor.close()



def get_item_data(data_num):
    file_path_user = '/opt/ml/server/winery/server/data/train_data.user'
    train_data_user = pd.read_csv(file_path_user, sep='\t')
    
    df = train_data_user[:data_num]


    # id,email,password,wine_list,mbti_result
    #user_column = ['user_id:token', 'count:float']
    user_column = ['count:float']
    # EX
    rename_rule = {
        #'user_id:token' : 'user_id',
        'count:float': 'password',
    }

    df = df[user_column]
    df = df.rename(columns=rename_rule)
    df['email'] = df['password'].copy
    df['wine_list'] = None
    df['mbti_result'] = None
    for idx in df.index:
        df['email'][idx] = f"user{idx}@example.com"
        df['wine_list'][idx] = [1070,6426,8087,11462,374,3060,11342,10967,6698,1696]
        df['mbti_result'][idx] = [1,2,3,4,5]

        
    return df



def insert_user_data(df,db=connection):
    insert_query = """
    INSERT INTO "user" (id,email,password,wine_list,mbti_result)
    VALUES (%s, %s, %s, %s, %s)
    """
    print("------- Making user table --------")
    with db.cursor() as cur, tqdm(total=df.shape[0], desc="Inserting data") as pbar:
        for _, row in df.iterrows():
            #print(row["price"])
            data = {
                'id': str(uuid4()),
                #'user_id' : row['user_id'],
                'email': str(row['email']),
                'password': str(row['password']),
                'wine_list': row['wine_list'], 
                'mbti_result' : row['mbti_result']
            }
            values = tuple(data.values())
            #pdb.set_trace()
            cur.execute(insert_query, values)
            pbar.update(1)

    db.commit()
    cur.close()



def main(args):
        # db 정보 받아오기
    conn = database.get_conn()

    # Check if user table is empty
    with conn.cursor() as cur: # 데이터 조작을 위한 인스턴스 생성
        table_name = "user"
        query = f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}');"
        cur.execute(query) #SQL 명령을 실행하기 위해 execut 함수 사용
        result = cur.fetchone()[0]

        if result:
            answer =input("table이 이미 생성되어 있습니다. 기존의 table을 지우시겠습니까 / yes or pass")
            if answer == "yes":
                delete_table(cur,table_name)

    # generate user db
    models.create_user_table(conn)
    
    # Check if user table is empty
    with conn.cursor() as cur: # 데이터 조작을 위한 인스턴스 생성
        cur.execute("SELECT EXISTS (SELECT 1 FROM public.user)") #SQL 명령을 실행하기 위해 execut 함수 사용
        is_empty = cur.fetchone()[0]

        if is_empty:
            print("Warnning : user table already contains data.")

    df = get_item_data(args.data_num)

    #DB에 data넣기
    insert_user_data(df,conn)


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--data_num", default='100', type=int)
    parser.add_argument("--model_output_path", default='/opt/ml/winery/code/text/models/model_output', type=str)
    parser.add_argument("--data", default='/opt/ml/winery/data/text_with_notelabel.csv', type=str)

     
    args = parser.parse_args()
    
    main(args)
    # if not os.path.exists(args.model_output_path):
    #     os.makedirs(args.model_output_path)
    



