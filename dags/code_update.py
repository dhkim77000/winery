from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta  # timedelta 임포트 수정
from github import Github


# Airflow DAG 설정
default_args = {
    'owner': 'recommy',
    'depends_on_past': False,
    'start_date': datetime(2023, 7, 1),  # DAG의 시작 날짜 설정
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# GitHub API Token
GITHUB_TOKEN = """-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACCsfAJ9bcXZG7BXpbprrViY/XJTIpFdcGlHR5pg7Uy7JQAAAKAuricFLq4n
BQAAAAtzc2gtZWQyNTUxOQAAACCsfAJ9bcXZG7BXpbprrViY/XJTIpFdcGlHR5pg7Uy7JQ
AAAEDb+pPvnQfhbbWOjbPnXA9pPx7KQ04gF5bImPz2lKRVxqx8An1txdkbsFelumutWJj9
clMikV1waUdHmmDtTLslAAAAGWtpbXlvdW5nc2VvMDMzMEBnbWFpbC5jb20BAgME
-----END OPENSSH PRIVATE KEY-----
"""

# GitHub Repository Information
REPO_OWNER = 'dhkim77000'
REPO_NAME = 'winery'
BRANCH_NAME = 'makeapi'

def update_github_branch():
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(f'{REPO_OWNER}/{REPO_NAME}')
    branch = repo.get_branch(BRANCH_NAME)

    # 브랜치 업데이트 로직 구현
    # 예를 들어, 브랜치 내용을 가져와서 원하는 처리를 수행하는 코드를 작성합니다.
    # 이 예시에서는 간단하게 브랜치 이름과 마지막 커밋 메시지를 출력하는 예제입니다.
    print(f'Branch Name: {branch.name}')
    print(f'Last Commit Message: {branch.commit.commit.message}')


## 확인된 dag 버킷 보내는 것 까지 완료 
with DAG(
    'update_github_branch',
    default_args=default_args,
    description='Update GitHub branch contents weekly',
    schedule_interval=timedelta(days=7),
) as dag:

    pull_task = BashOperator(
        task_id='git_pull',
        bash_command='cd /opt/ml/server/winery && git fetch && git pull origin makeapi',  # 명령어 구문 수정
        depends_on_past=True
    )

    pull_task


## mongo db interaction csv에 저장하고 bucket 보내는 것까지 -> pandas에서 오류
with DAG(
    'update_github_branch',
    default_args=default_args,
    description='Update GitHub branch contents weekly',
    schedule_interval=timedelta(days=7),  # 매주 실행
)as dag:
# Airflow Task 정의
    update_github_branch_task = PythonOperator(
        task_id='update_github_branch_task',
        python_callable=update_github_branch
    )

    pull_task = BashOperator(
        task_id='git_pull',
        bash_command = 'cd /opt/ml/server/winery & git branch & git fetch & git pull origin makeapi',
        depends_on_past = True
    )
# # DAG Task Dependency 설정
    update_github_branch_task >> pull_task