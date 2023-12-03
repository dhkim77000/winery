# hello_world.py
from datetime import timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.python_operator import PythonVirtualenvOperator




with DAG(
    dag_id="train_infer", # DAG의 식별자용 아이디입니다.
    description="train model and get inference", # DAG에 대해 설명합니다.
    start_date=days_ago(7), # DAG 정의 기준 2일 전부터 시작합니다.
    schedule_interval="0 0 * * *", # 매일 06:00에 실행합니다.
    tags=["model"],
) as dag:

    t0 = BashOperator(
    task_id="credential",
    bash_command='export GOOGLE_APPLICATION_CREDENTIALS="/home/dhkim/server_front/winery_AI/winery/airflow/polished-cocoa-404816-b981d3d391d9.json"',
    dag=dag,
    )

    t1 = BashOperator(
    task_id="preprocess",
    bash_command="python /home/dhkim/server_front/winery_AI/winery/code/data/prepare_data.py",
    dag=dag,
    )
 
    t2 = BashOperator(
    task_id="train",
    bash_command="python /home/dhkim/server_front/winery_AI/winery/Recbole/train.py --model_name DCN",
    dag=dag,
    )

    t3 = BashOperator(
    task_id="inference",
    bash_command="python /home/dhkim/server_front/winery_AI/winery/Recbole/inference.py",
    dag=dag,
    )
    # 테스크 순서를 정합니다.
    # t1 실행 후 t2를 실행합니다.
    t0 >> t1 >> t2 >> t3