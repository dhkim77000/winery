from joblib import Parallel, delayed
from database import get_mongo_db
import pymongo
import pdb
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import argparse
import os
import json
from database import get_db
from crud import update_wine_list_by_email

from google.cloud import storage

from utils import (read_last_date, 
                   write_ETL_log, 
                   parallel_push, 
                   get_GBQ_client, 
                   get_GBQ_table_ref,
                   push_data_GBQ)


def get_inter_update(collection, log_path):
    current_time = datetime.utcnow()
    current_time_unix = int(current_time.timestamp())
    
    last_time_unix = read_last_date(log_path)

    query = {"timestamp": {"$gte": last_time_unix, "$lt": current_time_unix}}

    cursor = collection.find(query)

    wine_ids = []
    emails = []
    timestamps = []
    ratings = []

    for document in tqdm(cursor):
        wine_ids.append(document['wine_id'])
        emails.append(document['email'])
        timestamps.append(document['timestamp'])
        ratings.append(document['rating'])

    update = pd.DataFrame({"timestamp":timestamps,
                       "email":emails,
                       "wine_id":wine_ids,
                       "rating":ratings})


    write_ETL_log(log_path, current_time, current_time_unix)

    return update 

def get_data_from_bucket():

    storage_client = storage.Client()

    bucket_name = 'recommendation_update'    

    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs())
    # 가장 최근에 업로드된 객체 찾기
    latest_blob = max(blobs, key=lambda x: x.time_created)

    destination_folder = '/home/dhkim/server_front/winery_server'    
    
    update = bucket.blob(latest_blob)

    destination_path = os.path.join(destination_folder, 'update.json')

    # 파일 다운로드
    update.download_to_filename(destination_path)


def update_recommendation():
    get_data_from_bucket()
    with open("/home/dhkim/server_front/winery_server/update.json", 'r') as f:
        updates = json.load(f)

    db = get_db()

    for email in updates:
        wine_list = updates[email]
        update_wine_list_by_email(db, email, wine_list)



def push_logs():
    
    log_path = '/home/dhkim/server_front/winery_server/server/ETL_log.txt'
    
    collection = get_mongo_db().interaction
    update = get_inter_update(collection, log_path)
    print(f"Updating {len(update)} interactions")

    #if args.parallel:
    #   num_cpu = os.cpu_count()
    #    parallel_push(update, num_cpu)
    
    client = get_GBQ_client()
    table_ref = get_GBQ_table_ref('inter', 'user_item')
    push_data_GBQ(client, table_ref, update)


def main(args):
    if args.mode =='push':
        push_logs()

    elif args.mode == 'update':
        update_recommendation()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='push', type=str)
    args = parser.parse_args()

    main(args)
    
