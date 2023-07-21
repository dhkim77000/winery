import pandas as pd
import numpy as np
import joblib
import ast
from joblib import Parallel, delayed
import pandas as pd
from pandas import DataFrame
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
from collections import defaultdict, deque
import re

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
        cols = json.load(f)
        for col in cols:
            if col in df.columns: df[col] = df[col].fillna('Empty')

    with open('/opt/ml/wine/code/data/meta_data/dict_columns.json','r',encoding='utf-8') as f:  
        cols = json.load(f)
        for col in cols:
            if col in df.columns: df[col] = df[col].fillna('{}')

    with open('/opt/ml/wine/code/data/meta_data/seq_columns.json','r',encoding='utf-8') as f:  
        cols = json.load(f)
        for col in cols:
            if col in df.columns: df[col] = df[col].fillna("[]")

    with open('/opt/ml/wine/code/data/meta_data/float_columns.json','r',encoding='utf-8') as f:  
        cols = json.load(f)
        #col = [c for c in col if '_count' in c]
        for col in cols:
            if col in df.columns: df[c] = df[c].fillna(0)
    

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
    with open('/opt/ml/wine/code/data/meta_data/string_columns.json','r',encoding='utf-8') as f:  
        single_category_columns = json.load(f)

    for c in single_category_columns:
        if c in df.columns: feature_mapper(df, c)
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

def item_preprocess(df, args):
    df = fill_na(df)
    df = drop_columns(df)
    map_all_single_features(df)
    df = map_all_list_features(df, args)
    #df = expand_notes(df, args)
    df['vectors'] = df['vectors'].apply(lambda x: " ".join(x))
    return df

def inter_preprocess(df, args):
    
    df = df[df['uid'].isna()== False]
    tqdm.pandas()

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


    if 'uid' in result.columns:

        urls = result['uid'].unique()
        user2idx = {v:k for k,v in enumerate(urls)}
        idx2user = {k:v for k,v in enumerate(urls)}

        with open(f'/opt/ml/wine/code/data/feature_map/user2idx.json','w',encoding='utf-8') as f:  
            json.dump(user2idx, f, ensure_ascii=False)
        with open(f'/opt/ml/wine/code/data/feature_map/idx2user.json','w',encoding='utf-8') as f:  
            json.dump(idx2user, f, ensure_ascii=False)

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

def keep_only_english(text):
    try:
        english_text = re.sub(r'[^a-zA-Z\s]', '', text)
        return english_text.lower().strip()
    except: return None

def find_vectors(columns_name : deque, grouped_vectors: DataFrame):
    column, name = columns_name.popleft()
    if name is not None:
        grouped_vectors = grouped_vectors.query(f"{column} == '{name}'")
        if columns_name:
            vector = find_vectors(columns_name, grouped_vectors.drop(column,axis = 1))
        else: 
            vector = grouped_vectors.vectors.mean()
            
        return vector
    else:
        return grouped_vectors.vectors.mean()
    
def get_item_vector(df, vector_path):
    with open(vector_path,'r') as f: 
        vectors = json.load(f)
    vector_list = []

    for id in tqdm(df.item_id):
        id = str(id)
        if id in vectors.keys():
            vector_list.append(np.array(vectors[id]))
        else:
            vector_list.append(None)
    df['vectors'] = vector_list
    return df

def fill_vectors(df : DataFrame, vector_path: str):
    df = get_item_vector(df, vector_path)

    for col in ['country','region', 'winetype', 'wine_style']:
        df[col] = df[col].apply(keep_only_english)
    
    vector_item = df[df.vectors.isna()==False]

    grouped_vectors = vector_item.groupby([
        'country',
        'region', 
        'winetype',
        'wine_style'
    ]).agg({'vectors': 'mean'}).reset_index()

    non_vectors = df[df['vectors'].isna()==True]
    non_vectors_cols = list(grouped_vectors.columns)
    non_vectors_cols.append('item_id')
    non_vectors = non_vectors.loc[:, non_vectors_cols]

    vectors = []
    for index, row in tqdm(non_vectors.iterrows()):
        columns_name = deque()

        for column, name in zip(row.keys(), row.values):
            if column != 'vectors':
                columns_name.append((column, name))
            else: break
        vector = find_vectors(columns_name, grouped_vectors)
        vectors.append(vector)
    
    non_vectors['vectors'] = vectors
    not_filled = non_vectors[non_vectors['vectors'].isna()==True]
    filled = non_vectors[non_vectors['vectors'].isna()==False]
    mean_vector = filled['vectors'].mean()
    
    not_filled['vectors'] = [ mean_vector for _ in range(len(not_filled))]
    filled_total = pd.concat([filled, not_filled], axis=0)

    no_vectors = df[df['vectors'].isna()==True]
    no_vectors.drop('vectors',axis = 1, inplace = True)
    yes_vectors = df[df['vectors'].isna()==False]

    no_vectors.reset_index(drop = True, inplace = True)
    filled_total.reset_index(drop = True, inplace = True)
    no_vectors = no_vectors.sort_values(by='item_id', ascending=False)
    filled_total = filled_total.sort_values(by='item_id', ascending=False)
    no_vectors.set_index('item_id', inplace = True)
    no_vectors['item_id'] = no_vectors.index
    yes_vectors['item_id'] = yes_vectors.index

    filled_total.set_index('item_id', inplace = True)
    filled_total['item_id'] = filled_total.index

    no_vectors['vectors'] = filled_total['vectors']
    no_vectors.reset_index(drop = True, inplace = True)
    yes_vectors.reset_index(drop = True, inplace = True)

    item_data_with_vectors = pd.concat([no_vectors,yes_vectors], axis=0)

    item_data_with_vectors = item_data_with_vectors.sort_values(by='item_id', ascending=True).reset_index(drop = True)
    return item_data_with_vectors

import pandas as pd


##data_to_normal(data,'email','timestamp','rating','wine_id')
def data_to_normal(data,user_id,timestamp,rating,wine_id):
    grouped_data = data.groupby(user_id)[rating].agg(['mean', 'std','count'])

    # 여러개 구매한 유저
    other_user = grouped_data[(grouped_data['count']>=5) & (grouped_data['std'] != 0)]

    other_user

    other_userlist = list(other_user.index)

    other_user_data = data[data[user_id].isin(other_userlist)].sort_values(by=user_id)
    other_user_data

    other_data = pd.merge(other_user_data,other_user, on =user_id,how='left')

    other_data = other_data.set_index(other_user_data.index)

    other_data['scaled_rating'] = (other_data[rating]-other_data['mean'])/other_data['std']
    print(other_data['scaled_rating'].quantile(0.75))
    result = other_data[['_id',user_id,timestamp,'scaled_rating',wine_id]]
    result.rename(columns = {'scaled_rating':'rating'})
    return result


