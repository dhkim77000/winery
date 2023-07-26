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

    with open('/opt/ml/wine/code/data/meta_data/seq_columns.json','r',encoding='utf-8') as f:  
        seq_columns = json.load(f)
    with open('/opt/ml/wine/code/data/meta_data/token_columns.json','r',encoding='utf-8') as f:  
        tok_columns = json.load(f)
    with open('/opt/ml/wine/code/data/meta_data/float_columns.json','r',encoding='utf-8') as f:  
        float_columns = json.load(f)



    item_data_cols = [
        'wine_id',

        'house', 

        'country', 'region1', 
        'winetype',  'wine_style', 'grape','vintage',

       'price', 'rating', 'num_votes', 
       'pairing',

       'Red Fruit', 'Tropical', 'Tree Fruit', 'Oaky', 'Ageing', 'Black Fruit',
       'Citrus', 'Dried Fruit', 'Earthy', 'Floral', 'Microbio', 'Spices', 'Vegetal', 

       'Light', 'Bold', 
       'Smooth', 'Tannic', 
       'Dry', 'Sweet',
       'Soft','Acidic',
       'Fizzy', 'Gentle']

    item_data = pd.read_csv('/opt/ml/wine/data/item_data_final.csv',
                            encoding= 'utf-8-sig',
                            usecols = item_data_cols)
    
    item_data.rename(columns = {'rating':'wine_rating'}, inplace= True)
    item_data = item_data.dropna(subset=['wine_id'], axis=0)

    item_data['wine_id'] = item_data['wine_id'].astype(int).astype('category')

    inter = pd.read_csv('/opt/ml/wine/data/inter_sample.csv', 
                                      encoding='utf-8-sig',
                                      usecols = ['email','rating','timestamp','wine_id'])
    
    inter = inter.dropna(subset=['wine_id'], axis=0)
    inter['wine_id'] = inter['wine_id'].astype(int).astype('category')
    inter = inter[inter['wine_id'].isin(item_data['wine_id'])]

    users = list(inter['email'].unique())
    users.sort()
    user2idx = {feature: index for index, feature in enumerate(users)}
    idx2user = {index: feature for index, feature in enumerate(users)}
    with open('/opt/ml/wine/code/data/feature_map/user2idx.json','w') as f: json.dump(user2idx,f)
    with open('/opt/ml/wine/code/data/feature_map/idx2user.json','w') as f: json.dump(idx2user,f)

    item_data = parallel(item_preprocess, item_data, args, num_cpu)
    inter = parallel(inter_preprocess, inter, args, num_cpu)
    inter = data_to_normal(inter,'email','timestamp','rating','wine_id')

    item_data.drop_duplicates(subset='wine_id', keep='first', inplace=True)
    columns_to_check = item_data.columns.drop('wine_id')
    item_data.dropna(subset=columns_to_check, how='all', inplace=True)

    


    if args.with_vector == 1:
        item_data = fill_vectors(item_data, args.wine_vector)
        
        wine_vectors = []
        for vector in item_data['vectors']: 
            wine_vectors.append(vector)
        wine_vectors = np.array(wine_vectors)

        item_data.set_index('wine_id', inplace= True)
        item_data['wine_id'] = item_data.index

        wine_ids = list(item_data['wine_id'])
        vector_dimension = wine_vectors.shape[1]

        index = faiss.IndexFlatIP(vector_dimension)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(wine_vectors, wine_ids)

        for id in tqdm(item_data.index):
            item_data = find_most_sim_item(item_data, id, index)

        #item_data['vectors'] = item_data['vectors'].apply(lambda x : x.tolist())
        
        print(item_data.isnull().sum())

    item2idx, user2idx, idx2item, idx2user = load_index_file()

    item_data['vectors'] = item_data['vectors'].apply(lambda x: " ".join(map(str, x))).str.replace('[', '').str.replace(']', '')

    inter.drop_duplicates(inplace = True)

    if args.expand_notes:
        item_data.to_csv('/opt/ml/wine/data/item_data_expand.csv', encoding='utf-8-sig', index=False)
    else:
        item_data.to_csv('/opt/ml/wine/data/item_data.csv', encoding='utf-8-sig', index=False)

    inter.to_csv('/opt/ml/wine/data/inter.csv', encoding='utf-8-sig', index=False)



    user_data = inter.groupby('email').agg(count=('scaled_rating', 'count'), mean=('scaled_rating', 'mean')).reset_index()

    print(f'Total {len(item_data)} items, {len(user_data)} users, {len(inter)} interactions')

    inter.rename(columns={'scaled_rating': 'user_rating'}, inplace=True)

    item_data.reset_index(drop = True, inplace = True)
    train_rating = pd.merge(inter.loc[:,['email','user_rating','timestamp','wine_id']],item_data.loc[:, 'wine_id'],on = 'wine_id', how = 'inner')

    train_rating.to_csv('/opt/ml/wine/data/train_rating.csv', encoding='utf-8', index=False)
    user_data.to_csv('/opt/ml/wine/data/user_data.csv', encoding='utf-8', index=False)

    return train_rating, user_data, item_data

def load_data_file():
    data_path = '/opt/ml/wine/data'
    
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
    
    with open('/opt/ml/wine/code/data/meta_data/seq_columns.json','r',encoding='utf-8') as f:  
        seq_columns = json.load(f)
    with open('/opt/ml/wine/code/data/meta_data/token_columns.json','r',encoding='utf-8') as f:  
        tok_columns = json.load(f)
    with open('/opt/ml/wine/code/data/meta_data/float_columns.json','r',encoding='utf-8') as f:  
        float_columns = json.load(f)
        
    dataset_name = 'train_data'
    # train_data 컬럼명 변경
    item2idx, user2idx, idx2item, idx2user = load_index_file()
    
    train_data['wine_id'] = train_data['wine_id'].astype(int).astype('category')
    train_data['email'] = train_data['email'].map(user2idx)
    train_data['email'] = train_data['email'].astype(int).astype('category')

    item_data['wine_id'] = item_data['wine_id'].astype(int).astype('category')


    
    user_data['email'] = user_data['email'].map(user2idx)
    user_data['email'] = user_data['email'].astype(int).astype('category')

    train_data.columns = to_recbole_columns(train_data.columns)
    user_data.columns = to_recbole_columns(user_data.columns)
    item_data.columns = to_recbole_columns(item_data.columns)
    
    # to_csv
    outpath = f"/opt/ml/wine/dataset/{dataset_name}"
    os.makedirs(outpath, exist_ok=True)

    columns_with_nan = item_data.columns[item_data.isnull().any()].tolist()
    for col in columns_with_nan:
        if col.split(':')[0] in tok_columns:
            item_data[col].fillna(item_data[col].mode().iloc[0], inplace= True)
            item_data[col] = item_data[col].replace('', 'other')
        elif col.split(':')[0] in seq_columns:
            item_data[col].fillna(item_data[col].mode().iloc[0], inplace= True)
            item_data[col] = item_data[col].replace('', 'other')
        elif col.split(':')[0] in float_columns:
            item_data[col] = item_data[col].fillna(item_data[col].mean())

    item_emb = item_data.loc[:,['wine_id:token','vectors:float_seq']]
    item_emb = item_emb.rename(columns = {'wine_id:token':'wid:token'})
    
    item_emb.to_csv(os.path.join(outpath,"train_data.itememb"),sep='\t',index=False, encoding='utf-8')
    train_data.to_csv(os.path.join(outpath,"train_data.inter"),sep='\t',index=False, encoding='utf-8')
    item_data.to_csv(os.path.join(outpath,"train_data.item"),sep='\t',index=False, encoding='utf-8')
    user_data.to_csv(os.path.join(outpath,"train_data.user"),sep='\t',index=False, encoding='utf-8')
    print(train_data.isnull().sum())
    print(item_data.isnull().sum())
    print(user_data.isnull().sum())

    print(train_data['wine_id:token'].nunique())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--expand_notes", default=False, type=bool)
    parser.add_argument("--with_vector", default=1, type=int)
    parser.add_argument("--wine_vector", default='/opt/ml/wine/data/emb_bert.json', type = str)
    parser.add_argument("--prepare_recbole", default=True, type=bool)
    args = parser.parse_args()


    train_data, user_data, item_data = prepare_dataset(args)
    if args.prepare_recbole:
        prepare_recbole_dataset()
