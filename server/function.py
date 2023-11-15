from psycopg2.extras import execute_values, register_uuid
from psycopg2.extensions import connection
import database , models
import pdb
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import json
import pandas as pd

def handle_nan(value):
    return None if isinstance(value, float) and math.isnan(value) else value

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



def sort_wine_by_distance(data):
    sorted_wine = sorted(data, key=lambda x: x[1], reverse=True)
    top_10 = [x[0] for x in sorted_wine[:10]]
    return top_10

#wine_list = sort_wine_by_distance(search_result)

# @router.post("/", status_code=status.HTTP_303_SEE_OTHER)
# async def user_login(request: Request,
#                      user = User,
#                      db: connection = Depends(get_conn)):
    
#     # Check if the user exists in the database
#     user = await get_user(db=db, email=email)
#     if user and verify_password(password, user.password):
#         # User exists and password is correct
#         # make access token
#         data = {
#             "sub": str(user.id),
#             "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#         }
#         access_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

#         response_data =  {
#             "access_token": access_token,
#             "token_type": "bearer",
#             "uid": str(user.id)
#             }
        
#         return response_data
#     else:
#         # User does not exist or password is incorrect
#         return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    

     
def wine2dic(result):
    data = {
        'id': result[0],
        'item_id': result[1],
        'winetype': result[2],
        'Red_Fruit': result[3],
        'Tropical': result[4],
        'Tree_Fruit': result[5],
        'Oaky': result[6],
        'Ageing': result[7],
        'Black_Fruit': result[8],
        'Citrus': result[9],
        'Dried_Fruit': result[10],
        'Earthy': result[11],
        'Floral': result[12],
        'Microbio': result[13],
        'Spices': result[14],
        'Vegetal': result[15],
        'Light': result[16],
        'Bold': result[17],
        'Smooth': result[18],
        'Tannic': result[19],
        'Dry': result[20],
        'Sweet': result[21],
        'Soft': result[22],
        'Acidic': result[23],
        'Fizzy': result[24],
        'Gentle': result[25],
        'vintage': result[26],
        'price': result[27],
        'wine_rating': result[28],
        'b_rating': result[29],
        'num_votes': result[30],
        'country': result[31],
        'region': result[32],
        'winery': result[33],
        'name': result[34],
        'wine_style': result[35],
        'house': result[36],
        'grape': result[37],
        'pairing': result[38]
    }
    return 


def bayesian_adj_rating(df):
    min_votes = 30

    tmp_df = df.dropna(subset=['rating', 'num_votes'])

    tmp_df['w_rating'] = tmp_df['rating'] * tmp_df['num_votes'] / (tmp_df['num_votes'].sum())
    bayesian_weight = 4  # Bayesian 가중치 (조절 가능)
    tmp_df['b_rating'] = (
        (tmp_df['num_votes'] / (tmp_df['num_votes'] + min_votes)) * tmp_df['rating'] +
        (min_votes / (tmp_df['num_votes'] + min_votes)) * tmp_df['rating'].mean()
    )
    df = pd.merge(tmp_df[['wine_id', 'b_rating']], df, how='right', on='wine_id')

    return df