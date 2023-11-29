from joblib import Parallel, delayed
from database import get_mongo_db
import pymongo
import pdb
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from google.cloud import storage
import os


def read_last_date(log_path):
    try:
        with open(log_path, 'r') as log_file:
            lines = log_file.readlines()
            if lines:
                last_line = lines[-1].strip()
                return last_line.split([1])
    except FileNotFoundError:
        print("Log file not found.")
    except Exception as e:
        print(f"Error reading log file: {e}")
    return -1


def write_ETL_log(log_path, time, unix_time):
    try:
        with open(log_path, 'a') as log_file:
            log_file.write(f"Execution: {unix_time} {time.strftime('%Y-%m-%d %H:%M')}\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")

def data_push(collection, log_path):
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

    df = pd.DataFrame({"timestamp":timestamps,
                       "email":emails,
                       "wine_id":wine_ids,
                       "ratings":ratings})
    old_df = get_inter_from_bucket()
    
    return df

def get_inter_from_bucket():
    bucket_name = 'input_features'    
    inter_data_source_blob_name = 'inter.csv'
    destination_folder = '/home/dhkim/server_front/winery_AI/winery/data'    

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    inter_blob = bucket.blob(inter_data_source_blob_name)

    file_path = os.path.join(destination_folder, inter_data_source_blob_name)
    inter_blob.download_to_filename(file_path)
    return pd.read_csv(file_path)


def main():
    log_path = '/home/dhkim/server_front/winery_server/server/ETL_log.txt'
    collection = get_mongo_db().interaction
    data_push(collection, log_path)

if __name__ == "__main__":
    main()
