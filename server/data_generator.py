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

from uuid import UUID, uuid4
import models, database, crud

def get_item_data():
    data_path = "/home/dhkim/server_front/winery_server/data"
    data = pd.read_csv(os.path.join(data_path, "item_data.csv"))
    # "grape"
    wine_column = ['wine_id','winetype','Red Fruit', 'Tropical', 'Tree Fruit', 'Oaky',\
        'Ageing', 'Black Fruit', 'Citrus', 'Dried Fruit', 'Earthy', 'Floral', \
        'Microbio','Spices', 'Vegetal', 'Light', 'Bold', 'Smooth', 'Tannic', 'Dry',\
        'Sweet', 'Soft', 'Acidic', 'Fizzy', 'Gentle','vintage','price',\
        'rating','num_votes','country','region1','winery','name','wine_style','house',\
        'grape', 'pairing']

    # EX
    rename_rule = {
        'Red Fruit' : 'Red_Fruit',
        'Tree Fruit': 'Tree_Fruit',
        'Black Fruit':'Black_Fruit',
        'Dried Fruit':'Dried_Fruit',

    }

    df = data[wine_column]
    df = data.rename(columns=rename_rule)
    df.loc[df['vintage'].isna(), 'vintage'] = 0
    df[['grape','pairing']] = df[['grape','pairing']].astype(str)
    df['grape'] = df['grape'].copy().apply(lambda x: x.split() if x is not None else [])
    df['pairing'] = df['pairing'].copy().apply(lambda x: x.split() if x is not None else [])
    #df['winetype'].fillna(0, inplace=True)
    # for data in range(df.shape[0]):
    #     df['rating'][data] = float(df['rating'][data])
    return df





def insert_wine_data(db=connection):
    insert_query = """
    INSERT INTO "wine" (id,item_id,winetype, Red_Fruit, Tropical, Tree_Fruit, Oaky, Ageing, Black_Fruit, Citrus, Dried_Fruit, Earthy, Floral, Microbio, Spices, Vegetal, Light, Bold, Smooth, Tannic, Dry, Sweet, Soft, Acidic, Fizzy, Gentle
    ,vintage,price,wine_rating,num_votes,country,region,winery,name,wine_style,house,grape,pairing)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s,%s, %s,%s,%s,%s,%s)
    """
    print("------- Making wine table --------")
    with db.cursor() as cur, tqdm(total=total_rows, desc="Inserting data") as pbar:
        for _, row in df.iterrows():
            #print(row["price"])
            data = {
                'id': str(uuid4()),
                'item_id' : int(row['wine_id']),
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
                'Gentle': int(row['Gentle']),
                'vintage': row['vintage'],
                'price': row['price'],
                'wine_rating': row['rating'],
                'num_votes': row['num_votes'],
                'country': str(row['country']),
                'region': str(row['region1']),
                'winery': str(row['winery']),
                'name': str(row['name']),
                'wine_style': str(row['wine_style']),
                'house': str(row['house']),
                'grape': row['grape'], 
                'pairing' : row['pairing'],
            }
            values = tuple(data.values())
            cur.execute(insert_query, values)
            pbar.update(1)
    db.commit()
    cur.close()

def create_mbti_data(db):
    models.create_mbiti_table(db)
    cur = db.cursor()
    cur.execute("SELECT EXISTS (SELECT 1 FROM mbti)")
    is_empty = cur.fetchone()[0]

    if is_empty:
        raise ValueError("MBTI table already contains data. Skipping insertion.")

    # CSV 파일에서 데이터 읽어오기
    with open(os.getcwd()+'/data/mbti_test.csv', 'r') as file:
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

    # generate wine db
    models.create_wine_table(conn)
    models.create_user_table(conn)
    models.create_mbiti_table(conn)
    
    # Check if wine table is empty
    with conn.cursor() as cur:
        cur.execute("SELECT EXISTS (SELECT 1 FROM wine)")
        is_empty = not cur.fetchone()[0]
        
        if is_empty:
            df = get_item_data()
            total_rows = df.shape[0]

            #DB에 data넣기
            insert_wine_data(conn)
            #create_mbti_data(conn)
        else:
            raise ValueError("Wine table already contains data. Skipping insertion.")
            
    
