import pandas as pd
import numpy as np
import joblib
import ast
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import joblib
import ast
from joblib import Parallel, delayed
import json
import os
from tqdm import tqdm
import os, pdb
from datetime import datetime
import pickle

def drop_columns(df):
    to_drop = ['Red Fruit','Tropical','Tree Fruit','Oaky',
               'Ageing','Black Fruit','Citrus','Dried Fruit','Earthy',
               'Floral','Microbio','Spices', 'Vegetal',
               'Unnamed: 58', 'None_child', 'None_count', "None",
                'Unnamed: 60', 'Unnamed: 61', 'Unnamed: 62', 'Unnamed: 63', 'Unnamed: 64']

    for c in to_drop:
        try:
            df.drop(c, axis = 1, inplace= True)  
        except Exception as e: 
            print(e)
    
    return df


def fill_na(df):
    with open('/opt/ml/wine/code/data/meta_data/string_columns.json','r',encoding='utf-8') as f:  
        col = json.load(f)
        df[col] = df[col].fillna('Empty')
    with open('/opt/ml/wine/code/data/meta_data/dict_columns.json','r',encoding='utf-8') as f:  
        col = json.load(f)
        df[col] = df[col].fillna('{}')
    with open('/opt/ml/wine/code/data/meta_data/seq_columns.json','r',encoding='utf-8') as f:  
        col = json.load(f)
        df[col] = df[col].fillna("[]")

    with open('/opt/ml/wine/code/data/meta_data/float_columns.json','r',encoding='utf-8') as f:  
        col = json.load(f)
        #col = [c for c in col if '_count' in c]
        for c in col:
            if c in df.columns: df[c] = df[c].fillna(0)
    

    return df


def str2list(x):
    if len(x) > 0:
        if x[0] != '[':
            list = [x]
        else: 
            list = ast.literal_eval(x)
            if len(list) == 0: list = ['Empty']
    else: list = ['Empty']

    return [str(s).replace(' ','') for s in list]


def feature_mapper(df, column):

    def space_remover(x):
        try:
            return str(x).replace(' ','_')
        except:
            print(column, x)
            return x
    df.loc[:,column] = df.loc[:,column].apply(lambda x: space_remover(x))

    unique_val = df[column].unique()
    feature2idx = {f:i for i, f in enumerate(unique_val)}
    idx2feature = {i:f for i, f in enumerate(unique_val)}

    if not os.path.exists('/opt/ml/wine/code/data/feature_map/'): 
        os.makedirs('/opt/ml/wine/code/data/feature_map/')

    with open(f'/opt/ml/wine/code/data/feature_map/{column}2idx.json','w',encoding='utf-8') as f:  
        json.dump(feature2idx, f, ensure_ascii=False)
    with open(f'/opt/ml/wine/code/data/feature_map/idx2{column}.json','w',encoding='utf-8') as f:  
        json.dump(idx2feature, f, ensure_ascii=False)

    return feature2idx, idx2feature

def list_feature_mapper(args, df, column):

    df[column] = df[column].apply(lambda x: str2list(x))
    exploded = df[column].explode(column)
    
    if args.prepare_recbole:
        df[column] = df[column].apply(lambda x: " ".join(x))

    unique_val = set(list(exploded))
    feature_dic = {}

    feature2idx = {f:i for i, f in enumerate(unique_val)}
    idx2feature = {i:f for i, f in enumerate(unique_val)}

    if not os.path.exists('/opt/ml/wine/code/data/feature_map/'): 
        os.makedirs('/opt/ml/wine/code/data/feature_map/')

    with open(f'/opt/ml/wine/code/data/feature_map/{column}2idx.json','w',encoding='utf-8') as f:  
        json.dump(feature2idx, f, ensure_ascii=False)
    with open(f'/opt/ml/wine/code/data/feature_map/idx2{column}.json','w',encoding='utf-8') as f:  
        json.dump(idx2feature, f, ensure_ascii=False)

    return df, feature2idx, idx2feature

def map_all_single_features(df):
    single_category_columns = ['country', 'region', 'winery', 'winetype', 'vintage', 'house', 'wine_style']
    for c in single_category_columns:
        feature_mapper(df, c)
    return  

def map_all_list_features(df,args):
    list_columns = ['grape','pairing']
    for c in list_columns:
        df ,_ ,_ = list_feature_mapper(args, df, c)
    return df 


def note_mapper(df, note_col):

    note = note_col

    note_col = note_col + '_child'
    note_col = note_col.replace(' ','_')
    
    try:
        df[note_col] = df[note_col].apply(lambda x: ast.literal_eval(x))
    except Exception as e:
        print(e)
    
    unique_val = []
    for note_dic in df[note_col]:
        unique_val.extend(list(note_dic.keys()))
    unique_val = list(set(unique_val))

    feature2idx = {f:i for i, f in enumerate(unique_val)}
    idx2feature = {i:f for i, f in enumerate(unique_val)}

    if not os.path.exists('/opt/ml/wine/code/data/feature_map/'): 
        os.makedirs('/opt/ml/wine/code/data/feature_map/')

    with open(f'/opt/ml/wine/code/data/feature_map/{note}2idx.json','w',encoding='utf-8') as f:  
        json.dump(feature2idx, f, ensure_ascii=False)
    with open(f'/opt/ml/wine/code/data/feature_map/idx2{note}.json','w',encoding='utf-8') as f:  
        json.dump(idx2feature, f, ensure_ascii=False)

    return feature2idx, idx2feature

def expand_notes(df, args):
    notes = ['Red Fruit','Tropical','Tree Fruit','Oaky',
        'Ageing','Black Fruit','Citrus','Dried Fruit','Earthy',
        'Floral','Microbio','Spices', 'Vegetal']
    
    i = 0
    if args.expand_notes:
        for note_col in tqdm(notes):

            note_df = []

            feature2idx, idx2feature = note_mapper(df, note_col)

            for total_count, note_dic in tqdm(zip(df[note_col+'_count'], df[note_col.replace(' ','_') + '_child'])):
                row_data = [0 for i in range(len(feature2idx))]

                if total_count != 0:
                    for note in note_dic:
                        row_data[feature2idx[note]] = note_dic[note] / total_count

                note_df.append(row_data)
            
            columns = [idx2feature[i] for i in range(len(idx2feature))]
            note_df = pd.DataFrame(note_df, columns=columns, index = df.index)

            if i == 0:
                result = note_df
                i += 1
            else:
                result = pd.concat([result, note_df], axis=1)

            df.drop(note_col.replace(' ','_') + '_child', axis = True, inplace = True)

        df = pd.concat([df, result], axis=1)
        
    else:
        for note_col in tqdm(notes):

            note_array = []

            feature2idx, idx2feature = note_mapper(df, note_col)

            for total_count, note_dic in tqdm(zip(df[note_col+'_count'], df[note_col.replace(' ','_') + '_child'])):
                row_data = [str(0) for i in range(len(feature2idx))]

                if total_count != 0:
                    for note in note_dic:
                        row_data[feature2idx[note]] = str(note_dic[note] / total_count)
                note_array.append(row_data)
            
            note_seq = note_col.replace(' ','_') + '_seq'
            df[note_seq] = note_array
            df[note_seq] = df[note_seq].apply(lambda x: " ".join(x))
            df.drop(note_col.replace(' ','_') + '_child', axis = 1, inplace = True)
    return df

def crawl_item_to_csv(df, args):
    df = fill_na(df)
    df = drop_columns(df)
    map_all_single_features(df)
    df = map_all_list_features(df, args)
    df = expand_notes(df, args)

    return df

def crawl_review_to_csv(df, args):
    
    df = df[df['user_url'].isna()== False]
    tqdm.pandas()
    df.loc[:,'date'] = df.loc[:,'date'].progress_apply(lambda x: pd.to_datetime(x))

    return df

def parallel(func, df, args, num_cpu):

    df_chunks = np.array_split(df, num_cpu)

    print('Parallelizing with ' +str(num_cpu)+'cores')
    with Parallel(n_jobs = num_cpu, backend="multiprocessing") as parallel:
        results = parallel(delayed(func)(df_chunks[i], args) for i in range(num_cpu))

    for i,data in enumerate(results):
        data.dropna(inplace = True)
        if i == 0:
            result = data
        else:
            result = pd.concat([result, data], axis = 0)

    for c in result.columns:
        if 'Unnamed:' in c:
            result.drop(c, axis = 1, inplace= True)  


    if 'user_url' in result.columns:

        urls = result['user_url'].unique()
        user2idx = {v:k for k,v in enumerate(urls)}
        idx2user = {k:v for k,v in enumerate(urls)}

        with open(f'/opt/ml/wine/code/data/feature_map/user2idx.json','w',encoding='utf-8') as f:  
            json.dump(user2idx, f, ensure_ascii=False)
        with open(f'/opt/ml/wine/code/data/feature_map/idx2user.json','w',encoding='utf-8') as f:  
            json.dump(idx2user, f, ensure_ascii=False)

    elif  'url' in result.columns:
        urls = result['url'].unique()
        item2idx = {v:k for k,v in enumerate(urls)}
        idx2item = {k:v for k,v in enumerate(urls)}

        with open(f'/opt/ml/wine/code/data/feature_map/item2idx.json','w',encoding='utf-8') as f:  
            json.dump(item2idx, f, ensure_ascii=False)
        with open(f'/opt/ml/wine/code/data/feature_map/idx2item.json','w',encoding='utf-8') as f:  
            json.dump(idx2item, f, ensure_ascii=False)

    return result

def to_recbole_columns(columns):

    meta_data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'meta_data')

    float_path = os.path.join(meta_data_folder, 'float_columns.json' )
    float_seq_path = os.path.join(meta_data_folder, 'float_seq_columns.json' )
    token_path = os.path.join(meta_data_folder, 'token_columns.json' )
    seq_path = os.path.join(meta_data_folder, 'seq_columns.json' )

    with open(float_path,'r',encoding='utf-8') as f:  float_columns = set(json.load(f))
    with open(float_seq_path,'r',encoding='utf-8') as f:  float_seq_columns = set(json.load(f))
    with open(token_path,'r',encoding='utf-8') as f:  token_columns = set(json.load(f))
    with open(seq_path,'r',encoding='utf-8') as f:  seq_columns = set(json.load(f))

    recbole_columns = []
    df = []
    for c in columns:
        if c in float_columns: recbole_columns.append(f'{c}:float')
        elif c in token_columns: recbole_columns.append(f'{c}:token')
        elif c in seq_columns: recbole_columns.append(f'{c}:token_seq')
        elif c in float_seq_columns: recbole_columns.append(f'{c}:float_seq')
        else:
            df.append(c)
            recbole_columns.append(f'{c}:float')
    

    print(f"{df[:5]}... are defaulely assigned to float type")

    return recbole_columns




def afterprocessing(sub,train):
    # 날짜를 datetime 형식으로 변환
    new_train = train.copy()
    new_train['time'] = new_train['time'].apply(lambda x: datetime.fromtimestamp(x))

    # 유저별 영화시청 마지막년도 추출
    user_mv_idx= new_train.groupby('user')['time'].max().reset_index()
    user_mv_idx['lastyear'] = user_mv_idx['time'].apply(lambda x : x.year)
    user_mv_idx.drop('time',inplace = True ,axis=1)

    # 영화 개봉년도와 유저시청년도 합친 데이터프레임 구축
    years = pd.read_csv("/opt/ml/input/data/train/years.tsv",sep = '\t')
    sub = pd.merge(sub,years, on = ['item'] , how = 'left')
    sub = pd.merge(sub,user_mv_idx,on =['user'],how ='left')

    # 늦게 개봉한 영화 제외하고 상위 10개 추출
    sub = sub[sub['lastyear'] >= sub['year']]
    sub = sub.groupby('user').head(10)[['user','item']]
    return sub



def feature_engineering():
    ### 추가
    return 

