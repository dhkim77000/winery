from psycopg2.extras import execute_values, register_uuid
from psycopg2.extensions import connection
import database , models


def get_top_10_items():
    # 데이터베이스 연결 설정
    conn = database.get_conn()
    cursor = conn.cursor()

    # SQL 쿼리 실행 (nan 값 제외)
    cursor.execute("SELECT item_id FROM wine WHERE NOT wine_rating IS NULL AND NOT num_votes IS NULL AND NOT wine_rating = 'nan' AND NOT num_votes = 'nan' ORDER BY wine_rating DESC, num_votes DESC LIMIT 10;")
    
    # 결과 가져오기
    result = cursor.fetchall()
    # 연결 종료
    cursor.close()
    conn.close()

    # 결과를 리스트로 반환
    item_ids = [row[0] for row in result]
    return item_ids


# 함수 호출 및 결과 출력
top_items = get_top_10_items()


def sort_wine_by_distance(data):
    sorted_wine = sorted(data, key=lambda x: x[1], reverse=True)
    top_10 = [x[0] for x in sorted_wine[:10]]
    return top_10




