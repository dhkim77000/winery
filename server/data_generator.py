from fastapi import FastAPI
from fastapi import FastAPI, Form, Request
from psycopg2.extras import execute_values, register_uuid
from psycopg2.extensions import connection
import pandas as pd
import os
import time
import csv
import sys
from uuid import UUID, uuid4
import models, database, crud
import pdb
from tqdm import tqdm

# Increase the field size limit
csv.field_size_limit(sys.maxsize)
data_path = "/opt/ml/api/server/data"
def get_item_data(data_path):
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

# ### CRUD create_wine module 세분화해서 가져오기 
# def insert_data(conn, data):
#     #pdb.set_trace()
#     data['id'] = uuid4() 
#     insert_row_query = f"""
#     INSERT INTO wine
#         (id,winetype, Red_Fruit , Tropical, Tree_Fruit, Oaky,
#         Ageing, Black_Fruit, Citrus, Dried_Fruit, Earthy, Floral, 
#         Microbio,Spices, Vegetal,Light, Bold, Smooth, Tannic,Dry,
#         Sweet,Soft,Acidic,Fizzy,Gentle)
#         VALUES (
#             '{data.id}',
#             '{data.winetype}',
#             {data.Red_Fruit},
#             {data['Tropical']},
#             {data['Tree_Fruit']},
#             {data['Oaky']},
#             {data['Ageing']},
#             {data['Black_Fruit']},
#             {data['Citrus']},
#             {data['Dried_Fruit']},
#             {data['Earthy']},
#             {data['Floral']},
#             {data['Microbio']},
#             {data['Spices']},
#             {data['Vegetal']},
#             {data['Light']},
#             {data['Bold']},
#             {data['Smooth']},
#             {data['Tannic']},
#             {data['Dry']},
#             {data['Sweet']},
#             {data['Soft']},
#             {data['Fizzy']},
#             {data['Acidic']},
#             {data['Gentle']}
#         );
#     """

#     with conn.cursor() as cur:
#         cur.execute(insert_row_query)
#         conn.commit()

# def generate_data(conn, df):
#     while len(df):
#         register_uuid()
#         insert_data(conn, df.sample(1).squeeze())

def create_wine_table(db: connection, data_path: str):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS "wine" (
        id UUID PRIMARY KEY NOT NULL UNIQUE,
        winetype VARCHAR(20),
        Red_Fruit int,
        Tropical int,
        Tree_Fruit int,
        Oaky int,
        Ageing int,
        Black_Fruit int,
        Citrus int,
        Dried_Fruit int,
        Earthy int,
        Floral int,
        Microbio int,
        Spices int,
        Vegetal int,
        Light int,
        Bold int,
        Smooth int,
        Tannic int,
        Dry int,
        Sweet int,
        Soft int,
        Acidic int,
        Fizzy int,
        Gentle int
    );
    """

    cur = db.cursor()
    cur.execute(create_table_query)
    db.commit()
    # Check if wine table is empty
    cur.execute("SELECT EXISTS (SELECT 1 FROM wine)")
    is_empty = cur.fetchone()[0]

    if is_empty:
        print("Wine table already contains data. Skipping insertion.")
        return

    df = get_item_data(data_path)
    total_rows = df.shape[0]

    insert_query = """
    INSERT INTO "wine" (id, winetype, Red_Fruit, Tropical, Tree_Fruit, Oaky, Ageing, Black_Fruit, Citrus, Dried_Fruit, Earthy, Floral, Microbio, Spices, Vegetal, Light, Bold, Smooth, Tannic, Dry, Sweet, Soft, Acidic, Fizzy, Gentle)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    print("------- Making wine table --------")
    with db.cursor() as cur, tqdm(total=total_rows, desc="Inserting data") as pbar:
        for _, row in df.iterrows():
            data = {
                'id': str(uuid4()),
                'winetype': row['winetype'],
                'Red_Fruit': int(row['Red_Fruit']),
                'Tropical': int(row['Tropical']),
                'Tree_Fruit': int(row['Tree_Fruit']),
                'Oaky': int(row['Oaky']),
                'Ageing': int(row['Ageing']),
                'Black_Fruit': int(row['Black_Fruit']),
                'Citrus': int(row['Citrus']),
                'Dried_Fruit': int(row['Dried_Fruit']),
                'Earthy': int(row['Earthy']),
                'Floral': int(row['Floral']),
                'Microbio': int(row['Microbio']),
                'Spices': int(row['Spices']),
                'Vegetal': int(row['Vegetal']),
                'Light': int(row['Light']),
                'Bold': int(row['Bold']),
                'Smooth': int(row['Smooth']),
                'Tannic': int(row['Tannic']),
                'Dry': int(row['Dry']),
                'Sweet': int(row['Sweet']),
                'Soft': int(row['Soft']),
                'Acidic': int(row['Acidic']),
                'Fizzy': int(row['Fizzy']),
                'Gentle': int(row['Gentle'])
            }
            values = tuple(data.values())
            cur.execute(insert_query, values)
            pbar.update(1)

    db.commit()
    cur.close()


def create_mbti_table(db: connection):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS "mbti" (
        mbti_id int PRIMARY KEY,
        item text[]
    );
    """
    cur = db.cursor()
    cur.execute(create_table_query)
    db.commit()
    cur.execute("SELECT EXISTS (SELECT 1 FROM mbti)")
    is_empty = cur.fetchone()[0]

    if is_empty:
        print("MBTI table already contains data. Skipping insertion.")
        return
    # CSV 파일에서 데이터 읽어오기
    with open('/opt/ml/api/server/data/mbti_test.csv', 'r') as file:
        csv_data = csv.reader(file)
        next(csv_data)

        insert_query = 'INSERT INTO "mbti" (mbti_id, item) VALUES (%s, %s)'

        total_rows = sum(1 for _ in csv_data)  # Count the total number of rows

        # Reset the file pointer to the beginning of the file
        file.seek(0)
        next(csv_data)  # Skip the header row
        
        print("------- Making mbti table --------")
        with tqdm(total=total_rows, desc="Inserting data") as pbar:
            for row in csv_data:
                mbti_id = int(row[0])  # mbti_id는 첫 번째 열의 값
                for item in row[1:]:  # 두 번째 열부터 마지막 열까지의 값을 순회하며 삽입
                    values = (mbti_id, item)
                    cur.execute(insert_query, values)
                pbar.update(1)

    db.commit()
    cur.close()



if __name__ == "__main__":

    # db 정보 받아오기
    conn = database.get_conn()

    # DB에 data넣기
    create_wine_table(conn,data_path)
    create_mbti_table(conn)


