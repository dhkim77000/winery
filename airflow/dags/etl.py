# hello_world.py
from datetime import timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.python_operator import PythonVirtualenvOperator




with DAG(
    dag_id="ETL", # DAG의 식별자용 아이디입니다.
    description="Update Data to Warehouse", # DAG에 대해 설명합니다.
    start_date=days_ago(7), # DAG 정의 기준 2일 전부터 시작합니다.
    schedule_interval="0 0 * * *", # 매일 06:00에 실행합니다.
    tags=["ETL"],
) as dag:

    t0 = BashOperator(
    task_id="credential",
    bash_command='export GOOGLE_APPLICATION_CREDENTIALS="/home/dhkim/server_front/winery_server/airflow/polished-cocoa-404816-b981d3d391d9.json"',
    dag=dag,
    )

    t1 = BashOperator(
    task_id="update",
    bash_command="python /home/dhkim/server_front/winery_server/server/etl.py --mode push",
    dag=dag,
    )
 
    t0 >> t1 