from google.cloud import storage
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import numpy as np
from joblib import Parallel, delayed
import pdb



def read_last_date(log_path):
    try:
        with open(log_path, 'r') as log_file:
            lines = log_file.readlines()
            if lines:
                last_line = lines[-1].strip()
                return int(last_line.split()[1])
    except FileNotFoundError:
        print("Log file not found.")
    except Exception as e:
        print(f"Error reading log file: {e}")
    return -1


def write_ETL_log(log_path, time, unix_time, num_update):
    try:
        with open(log_path, 'a') as log_file:
            log_file.write(f"Execution[{num_update}]: {unix_time} {time.strftime('%Y-%m-%d %H:%M')}, \n")
    except Exception as e:
        print(f"Error writing to log file: {e}")



def get_GBQ_client():
    KEY_PATH = "/home/dhkim/server_front/winery_AI/winery/airflow/polished-cocoa-404816-b981d3d391d9.json"
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    client = bigquery.Client(credentials = credentials, project = credentials.project_id)
    return client


def get_GBQ_table_ref(dataset_id, table_id):
    table_ref = get_GBQ_client().dataset(dataset_id).table(table_id)
    return table_ref

def push_data_GBQ(client, table_ref, df):
    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)

    print(job.result())


###안됨
def parallel_push(df, num_cpu):

    client = get_GBQ_client()
    table_ref = get_GBQ_table_ref('inter', 'user_item')

    df_chunks = np.array_split(df, num_cpu)

    print('Parallelizing with ' +str(num_cpu)+'cores')
    with Parallel(n_jobs = num_cpu, backend="threading") as parallel:
        parallel(delayed(push_data_GBQ)(client, table_ref, df_chunks[i]) for i in range(num_cpu))



    
