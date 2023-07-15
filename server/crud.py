from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from pydantic import BaseModel
from sqlalchemy import func
from passlib.context import CryptContext
from psycopg2.extras import execute_values, register_uuid
from schema import UserCreate, WinePost , Login_User
from models import User, Wine,MBTI, create_user_table,create_wine_table
import pdb , math
from fastapi.security import OAuth2PasswordRequestForm
from psycopg2.extensions import connection

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create SQLAlchemy engine and session
async def add_user_winelist(db: connection, user_id: int):
    with db.cursor() as cur:
        cur.execute("SELECT * FROM public.user WHERE user_id = %s", (user_id,))
        result = cur.fetchone()
    if result is None:
        raise HTTPException(status_code=404, detail=f"존재하지 않는 USER입니다.")

    else:
            
    # class User(Base):
    #     __tablename__ = "user"
    #     id = Column(UUID(as_uuid=True), primary_key=True, nullable=False, unique=True)
    #     email = Column(String, nullable=False, unique=True)
    #     password = Column(String, nullable=False)
    # 여기에 wine_list 컬럼 추가
        return


async def get_user(db: connection, email: str,):

    with db.cursor() as cur:
        cur.execute("SELECT * FROM public.user WHERE email = %s", (email,))
        result = cur.fetchone()

    if result is None:
        raise HTTPException(status_code=404, detail=f"존재하지 않는 이메일입니다.")
    else:
        user = Login_User(
            id=result[0],
            email=result[1],
            password=result[2],
            wine_list = result[3]
        )
        return user
    
async def get_mbti_data(db: connection, mbti_id):
    

    with db.cursor() as cur:
        cur.execute("SELECT * FROM mbti WHERE mbti_id = %s", (mbti_id,))
        result = cur.fetchone()

    if result is None:
        raise HTTPException(status_code=404, detail=f"결과가 없습니다.")
    else:
        mbti = MBTI(
            mbti_id=result[0],
            wine_list=result[1],
        )
        return mbti
    
async def get_wine_data(db: connection, wine_id):
    

    with db.cursor() as cur:
        cur.execute("SELECT * FROM wine WHERE item_id = %s", (wine_id,))
        result = cur.fetchone()

    if result is None:
        raise HTTPException(status_code=404, detail=f"존재하지 않는 와인입니다.")
    else:
        result = ['Null' if ((isinstance(value, float) and math.isnan(value)) or
                     (isinstance(value, str) and value.lower() == 'nan')) 
                  else value for value in result]
        wine = Wine(
            id = result[0],
            item_id = result[1],
            winetype = result[2],
            Red_Fruit = result[3],
            Tropical = result[4],
            Tree_Fruit = result[5],
            Oaky = result[6],
            Ageing = result[7],
            Black_Fruit = result[8],
            Citrus = result[9],
            Dried_Fruit = result[10],
            Earthy = result[11],
            Floral = result[12],
            Microbio = result[13],
            Spices = result[14],
            Vegetal = result[15],
            Light = result[16],
            Bold = result[17],
            Smooth = result[18],
            Tannic = result[19],
            Dry = result[20],
            Sweet = result[21],
            Soft = result[22],
            Acidic = result[23],
            Fizzy = result[24],
            Gentle = result[25],
            vintage = result[26],
            price = result[27],
            wine_rating = result[28],
            num_votes = result[29],
            country = result[30],
            region= result[31],
            winery= result[32],
            name= result[33],
            wine_style= result[34],
            house = result[35],
            grape = result[36],
            pairing = result[37],
        )
        return wine

async def get_wine_data_simple(db: connection, wine_id):
    

    with db.cursor() as cur:
        cur.execute("SELECT * FROM wine WHERE item_id = %s", (wine_id,))
        result = cur.fetchone()

    if result is None:
        raise HTTPException(status_code=404, detail=f"존재하지 않는 와인입니다.")
    else:
        result = ['Null' if ((isinstance(value, float) and math.isnan(value)) or
                     (isinstance(value, str) and value.lower() == 'nan')) 
                  else value for value in result]

        wine = Wine(
            id = result[0],
            item_id = result[1],
            winetype = result[2],
            vintage = result[26],
            price = result[27],
            wine_rating = result[28],
            num_votes = result[29],
            country = result[30],
            region= result[31],
            winery= result[32],
            name= result[33],
            wine_style= result[34],
            pairing = result[37],
        )
        return wine
    
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


async def check_email_exist(email, cur):

    cur.execute("SELECT * FROM public.user WHERE email = %s", (email,))
    result = cur.fetchone()

    if result is not None:
        return True
    else:
        return False
    

async def create_user(db: connection, user_create: UserCreate):
    create_user_table(db)
    register_uuid()
    db_user = User(id=user_create.id,
                   password=pwd_context.hash(user_create.password1),
                   email=user_create.email)
    insert_user_query = f"""
    INSERT INTO "{db_user.__tablename__}"
        (id, email, password)
        VALUES %s;
        """
    values = [(db_user.id, db_user.email, db_user.password)]
    
    # Execute the query
    with db.cursor() as cur:
        exist = await check_email_exist(user_create.email, cur)
        if exist:
            raise HTTPException(status_code=404, detail=f"이미 존재하는 이메일입니다.")
        else:
            print(insert_user_query)
            execute_values(cur, insert_user_query, values)
            db.commit()
    


async def add_mbti_feature(email, mbti_id, db: connection):


    # Check if 'mbti' and 'item' columns exist in the user table
    cursor = db.cursor()
    cursor.execute('SELECT column_name FROM information_schema.columns WHERE table_name = \'user\'')
    columns = [column[0] for column in cursor.fetchall()]
    if 'mbti' not in columns:
        cursor.execute("ALTER TABLE public.user ADD COLUMN mbti TEXT")
    if 'item' not in columns:
        cursor.execute("ALTER TABLE public.user ADD COLUMN item TEXT")
    db.commit()

    # Find the corresponding MBTI item
    cursor.execute('SELECT item FROM "mbti" WHERE mbti_id = %s', (mbti_id,))
    item = cursor.fetchone()

    if item:
        item = item[0]
        # Update the user table with the MBTI and item
        cursor.execute('UPDATE public.user SET mbti = %s, item = %s WHERE email = %s', (mbti_id, item, email))
        db.commit()
    else:
        print("Invalid mbti_id")

    # Close the database connection
    cursor.close()
    db.close()

    return item

# async def create_wine(db: connection, wine_get: WinePost):
#     create_wine_table(db)
#     register_uuid()
#     db_wine = Wine(id=wine_get.id,
#                    winetype=wine_get.winetype,
#                    Red_Fruit=wine_get.Red_Fruit,Tropical=wine_get.Tropical,
#                    Tree_Fruit=wine_get.Tree_Fruit,Oaky=wine_get.Oaky,
#                    Ageing=wine_get.Ageing,Black_Fruit=wine_get.Black_Fruit,
#                    Citrus=wine_get.Citrus,Dried_Fruit=wine_get.Dried_Fruit,
#                    Earthy=wine_get.Earthy,Floral=wine_get.Floral,
#                    Microbio=wine_get.Microbio,Spices=wine_get.Spices,
#                    Vegetal=wine_get.Vegetal,Light=wine_get.Light,
#                    Bold=wine_get.Bold,Smooth=wine_get.Smooth,
#                    Tannic=wine_get.Tannic,Dry=wine_get.Dry,
#                    Sweet=wine_get.Sweet,Soft=wine_get.Soft,
#                    Gentle=wine_get.Gentle,
#                    Acidic=wine_get.Acidic,Fizzy=wine_get.Fizzy,
#                    vintage = wine_get.vintage, price = wine_get.price,
#                    wine_rating = wine_get.wine_rating,num_votes = wine_get.num_votes,
#                    item_id = wine_get.item_id
#                    )


#     insert_wine_query = f"""
#     INSERT INTO "{db_wine.__tablename__}"
#         (id,winetype,Red_Fruit,Tropical,Tree_Fruit,,Oaky,Ageing,
#         Black_Fruit,Citrus,Dried_Fruit,Earthy,Floral,Microbio,Spices,Vegetal,Light,Bold,Smooth,Tannic,Dry,Sweet,Soft,Acidic,Fizzy,Gentle
#         vintage, price, wine_rating, num_votes)
#         VALUES %s;
#         """
#     values = [(db_wine.id,
#                    db_wine.winetype,
#                    db_wine.Red_Fruit,db_wine.Tropical,
#                    db_wine.Tree_Fruit,db_wine.Oaky,
#                    db_wine.Ageing,db_wine.Black_Fruit,
#                    db_wine.Citrus,db_wine.Dried_Fruit,
#                    db_wine.Earthy,db_wine.Floral,
#                    db_wine.Microbio,db_wine.Spices,
#                    db_wine.Vegetal,db_wine.Light,
#                    db_wine.Bold,db_wine.Smooth,
#                    db_wine.Tannic,db_wine.Dry,
#                    db_wine.Sweet,db_wine.Soft,
#                    db_wine.Gentle,
#                    db_wine.Acidic,db_wine.Fizzy,
#                    db_wine.vintage,db_wine.price,
#                    db_wine.wine_rating,db_wine.num_votes)]
#     print(insert_wine_query)
#     # Execute the query
#     with db.cursor() as cur:
#         execute_values(cur, insert_wine_query, values)
#         db.commit()
    
