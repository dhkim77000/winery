from utils import *

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
import argparse

def load_index_file():

    mapper_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'feature_map')

    item2idx_path = 'item2idx.json'
    item2idx_path = os.path.join(mapper_folder, item2idx_path)
    idx2item_path = 'idx2item.json'
    idx2item_path = os.path.join(mapper_folder, idx2item_path)

    user2idx_path = 'user2idx.json'
    user2idx_path = os.path.join(mapper_folder, user2idx_path)
    idx2user_path = 'idx2user.json'
    idx2user_path = os.path.join(mapper_folder, idx2user_path)

    try:
        with open(item2idx_path,'r',encoding='utf-8') as f:  item2idx = json.load(f)    
        with open(user2idx_path,'r',encoding='utf-8') as f:  user2idx = json.load(f)
        with open(idx2item_path,'r',encoding='utf-8') as f:  idx2item = json.load(f)
        with open(idx2user_path,'r',encoding='utf-8') as f:  idx2user = json.load(f)
    except:
        train_rating, item2idx, user2idx, idx2item, idx2user = prepare_dataset()
   

    return item2idx, user2idx, idx2item, idx2user

def prepare_dataset(args):
    num_cpu = os.cpu_count()
    item_data_cols = [
        'item_id',

        'house', 

        'country', 'region1', 
        'winetype',  'wine style', 'grape','vintage',

       'price', 'wine_rating', 'num_votes', 
       'pairing',

       'Red Fruit', 'Tropical', 'Tree Fruit', 'Oaky', 'Ageing', 'Black Fruit',
       'Citrus', 'Dried Fruit', 'Earthy', 'Floral', 'Microbio', 'Spices', 'Vegetal', 

       'Light', 'Bold', 
       'Smooth', 'Tannic', 
       'Dry', 'Sweet',
       'Soft','Acidic',
       'Fizzy', 'Gentle']
    
    item_data = pd.read_csv('/opt/ml/wine/data/item_df_allfeature.csv',
                            encoding= 'utf-8-sig',
                            usecols = item_data_cols)
    
    item_data['item_id'] = item_data['item_id'].astype(int).astype('category')
    
    item_data = fill_vectors(item_data, '/opt/ml/wine/data/wine_vector.json')
    wine_vectors = []

    for vector in item_data['vectors']: wine_vectors.append(vector)
    wine_vectors = np.array(wine_vectors)


    inter = pd.read_csv('/opt/ml/wine/data/review_df_total.csv', 
                                      encoding='utf-8-sig',
                                      usecols = ['uid','rating','date','item_id'])
    inter['item_id'] = inter['item_id'].astype(int).astype('category')
    inter['uid'] = inter['uid'].astype('category')
    
    inter = data_to_normal(inter,'email','timestamp','rating','wine_id')

    item_data = parallel(item_preprocess, item_data, args, num_cpu)
    inter = parallel(inter_preprocess, inter, args, num_cpu)
    
    item_data.drop_duplicates(subset='item_id', keep='first', inplace=True)
    columns_to_check = item_data.columns.drop('item_id')
    item_data.dropna(subset=columns_to_check, how='all', inplace=True)

    item2idx, user2idx, idx2item, idx2user = load_index_file()

    inter.drop_duplicates(inplace = True)
    ####추가
    if args.expand_notes:
        item_data.to_csv('/opt/ml/wine/data/item_data_expand.csv', encoding='utf-8-sig', index=False)
    else:
        item_data.to_csv('/opt/ml/wine/data/item_data.csv', encoding='utf-8-sig', index=False)
    inter.to_csv('/opt/ml/wine/data/inter.csv', encoding='utf-8-sig', index=False)



    user_data = inter.groupby('uid').agg(count=('rating', 'count'), mean=('rating', 'mean')).reset_index()

    print(f'Total {len(item_data)} items, {len(user_data)} users, {len(inter)} interactions')

    inter.rename(columns={'rating': 'user_rating','date': 'timestamp'}, inplace=True)


    train_rating = pd.merge(inter.loc[:,['uid','user_rating','timestamp','item_id']],
                            item_data.loc[:, 'item_id'],
                            on = 'item_id', how = 'inner')

    train_rating.to_csv('/opt/ml/wine/data/train_rating.csv', encoding='utf-8', index=False)
    user_data.to_csv('/opt/ml/wine/data/user_data.csv', encoding='utf-8', index=False)

    return train_rating, user_data, item_data

def load_data_file():

    data_path = '/opt/ml/wine/data'
    # train load
    try:
        train_data = pd.read_csv(os.path.join(data_path, 'train_rating.csv'), encoding = 'utf-8-sig')
        user_data = pd.read_csv(os.path.join(data_path, 'user_data.csv'), encoding = 'utf-8-sig')
        item_data = pd.read_csv(os.path.join(data_path, 'item_data.csv'), encoding = 'utf-8-sig')
    except:
        print('No files found, prepare dataste')
        train_data, user_data, item_data = prepare_dataset()

    return train_data, user_data, item_data

def prepare_recbole_dataset():
    train_data, user_data, item_data = load_data_file()
    save_atomic_file(train_data, user_data, item_data)

def save_atomic_file(train_data, user_data, item_data):
    dataset_name = 'train_data'
    # train_data 컬럼명 변경

    train_data['item_id'] = train_data['item_id'].astype(int).astype('category')
    train_data['user_id'] = train_data['user_id'].astype(int).astype('category')

    item_data['item_id'] = item_data['item_id'].astype(int).astype('category')
    
    user_data['user_id'] = user_data['user_id'].astype(int).astype('category')

    train_data.columns = to_recbole_columns(train_data.columns)
    user_data.columns = to_recbole_columns(user_data.columns)
    item_data.columns = to_recbole_columns(item_data.columns)
    
    # to_csv
    outpath = f"/opt/ml/winery/dataset/{dataset_name}"
    os.makedirs(outpath, exist_ok=True)
    import pandas as pd

    train_data.to_csv(os.path.join(outpath,"train_data.inter"),sep='\t',index=False, encoding='utf-8')
    item_data.to_csv(os.path.join(outpath,"train_data.item"),sep='\t',index=False, encoding='utf-8')
    user_data.to_csv(os.path.join(outpath,"train_data.user"),sep='\t',index=False, encoding='utf-8')
    print(train_data.isnull().sum())
    print(item_data.isnull().sum())
    print(user_data.isnull().sum())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--expand_notes", default=False, type=bool)
    parser.add_argument("--prepare_recbole", default=True, type=bool)
    args = parser.parse_args()


    train_data, user_data, item_data = prepare_dataset(args)
    if args.prepare_recbole:
        prepare_recbole_dataset()
