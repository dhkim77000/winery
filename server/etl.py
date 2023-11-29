from joblib import Parallel, delayed
from database import get_mongo_db
import pymongo
import pdb
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import argparse
import os
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




def main():
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--parallel", action="store_true", help="Whether to parallelize process"
    )
    args = parser.parse_args()
    main()
    
