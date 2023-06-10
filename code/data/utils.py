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

def drop_columns(df):
    to_drop = ['Red Fruit','Tropical','Tree Fruit','Oaky',
               'Ageing','Black Fruit','Citrus','Dried Fruit','Earthy',
               'Floral','Microbio','Spices', 'Vegetal',
               'Unnamed: 58', 'None_child', 'Unnamed: 60', 'Unnamed: 61', 
               'Unnamed: 62', 'Unnamed: 63', 'Unnamed: 64']
    
    for c in to_drop:
        try:
            df.drop(c, axis = 1, inplace= True)  
        except Exception as e: 
            print(e)
    
    return df


def fill_na(df):
    with open('/opt/ml/wine/code/data/meta_data/string_columns.json','r',encoding='utf-8') as f:  
        col = json.load(f)
        df[col] = df[col].fillna('')
    with open('/opt/ml/wine/code/data/meta_data/dict_columns.json','r',encoding='utf-8') as f:  
        col = json.load(f)
        df[col] = df[col].fillna('{}')
    with open('/opt/ml/wine/code/data/meta_data/list_columns.json','r',encoding='utf-8') as f:  
        col = json.load(f)
        df[col] = df[col].fillna('[]')
    return df


def str2list(x):
    if len(x) > 0:
        if x[0] != '[':
            list = [x]
        else: 
            list = ast.literal_eval(x)
    else: list = []
    
    return list

def feature_mapper(df, column):
    unique_val = df[column].unique()
    feature2idx = {f:i for i, f in enumerate(unique_val)}
    idx2feature = {i:f for i, f in enumerate(unique_val)}

    if not os.path.exists('/opt/ml/wine/code/meta_data/'): 
        os.makedirs('/opt/ml/wine/code/meta_data/')

    with open(f'/opt/ml/wine/code/feature_map/{column}2idx.json','w',encoding='utf-8') as f:  
        json.dump(feature2idx, f, ensure_ascii=False)
    with open(f'/opt/ml/wine/code/feature_map/idx2{column}.json','w',encoding='utf-8') as f:  
        json.dump(idx2feature, f, ensure_ascii=False)

    return feature2idx, idx2feature

def list_feature_mapper(df, column):

    df[column] = df[column].apply(lambda x: str2list(x))

    exploded = df[column].explode(column)
    unique_val = set(list(exploded))
    feature_dic = {}

    feature2idx = {f:i for i, f in enumerate(unique_val)}
    idx2feature = {i:f for i, f in enumerate(unique_val)}

    if not os.path.exists('/opt/ml/wine/code/meta_data/'): 
        os.makedirs('/opt/ml/wine/code/meta_data/')

    with open(f'/opt/ml/wine/code/feature_map/{column}2idx.json','w',encoding='utf-8') as f:  
        json.dump(feature2idx, f, ensure_ascii=False)
    with open(f'/opt/ml/wine/code/feature_map/idx2{column}.json','w',encoding='utf-8') as f:  
        json.dump(idx2feature, f, ensure_ascii=False)

    return df, feature2idx, idx2feature

def map_all_single_features(df):
    single_category_columns = ['country', 'region', 'winery', 'winetype', 'vintage', 'house', 'wine_style']
    for c in single_category_columns:
        feature_mapper(df, c)
    return  

def map_all_list_features(df):
    list_columns = ['grape','pairing']
    for c in list_columns:
        df ,_ ,_ = list_feature_mapper(df, c)
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

    if not os.path.exists('/opt/ml/wine/code/meta_data/'): 
        os.makedirs('/opt/ml/wine/code/meta_data/')

    with open(f'/opt/ml/wine/code/feature_map/{note}2idx.json','w',encoding='utf-8') as f:  
        json.dump(feature2idx, f, ensure_ascii=False)
    with open(f'/opt/ml/wine/code/feature_map/idx2{note}.json','w',encoding='utf-8') as f:  
        json.dump(idx2feature, f, ensure_ascii=False)

    return feature2idx, idx2feature

def expand_notes(df):
    notes = ['Red Fruit','Tropical','Tree Fruit','Oaky',
        'Ageing','Black Fruit','Citrus','Dried Fruit','Earthy',
        'Floral','Microbio','Spices', 'Vegetal']
    
    i = 0
    for note_col in tqdm(notes):

        note_df = []

        feature2idx, idx2feature = note_mapper(df, note_col)

        for note_dic in tqdm(df[note_col.replace(' ','_') + '_child']):
            row_data = [0 for i in range(len(feature2idx))]

            for note in note_dic:
                row_data[feature2idx[note]] = note_dic[note]
  
            note_df.append(row_data)
        
        columns = [idx2feature[i] for i in range(len(idx2feature))]
        note_df = pd.DataFrame(note_df, columns=columns)

        if i == 0:
            result = note_df
            i += 1
        else:
            result = pd.concat([result, note_df], axis=1)

        df.drop(note_col.replace(' ','_') + '_child', axis = True, inplace = True)

    df = pd.concat([df, result], axis=1)
    return df
  


def crawl_item_to_csv(df):
    df = fill_na(df)
    df = drop_columns(df)
    map_all_single_features(df)
    df = map_all_list_features(df)
    df = expand_notes(df)

    return df

def crawl_review_to_csv(df):
    
    df = df[df['user_url'].isna()== False]
    tqdm.pandas()
    df['date'] = df['date'].progress_apply(lambda x: pd.to_datetime(x))
    return df

def parallel(func, df, num_cpu):

    df_chunks = np.array_split(df, num_cpu)

    print('Parallelizing with ' +str(num_cpu)+'cores')
    with Parallel(n_jobs = num_cpu, backend="multiprocessing") as parallel:
        results = parallel(delayed(func)(df_chunks[i]) for i in range(num_cpu))

    for i,data in enumerate(results):
        if i == 0:
            result = data
        else:
            result = pd.concat([result, data], axis = 0)

    return result

if __name__ == '__main__':
    num_cpu = os.cpu_count()

    crawled_item_data = pd.read_csv('/opt/ml/wine/data/wine_df.csv')
    crawled_review_data = pd.read_csv('/opt/ml/wine/data/review_df.csv')

    item_data = parallel(crawl_item_to_csv, crawled_item_data,  num_cpu)
    review_data = parallel(crawl_review_to_csv, crawled_review_data,  num_cpu)

    item_data.to_csv('/opt/ml/wine/data/item_data.csv', encoding='utf-8-sig')
    review_data.to_csv('/opt/ml/wine/data/review_data.csv', encoding='utf-8-sig')