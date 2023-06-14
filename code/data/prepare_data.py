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

def prepare_dataset():
    num_cpu = os.cpu_count()

    crawled_item_data = pd.read_csv('/opt/ml/wine/data/wine_df.csv')
    crawled_review_data = pd.read_csv('/opt/ml/wine/data/review_df.csv')

    item_data = parallel(crawl_item_to_csv, crawled_item_data,  num_cpu)
    review_data = parallel(crawl_review_to_csv, crawled_review_data,  num_cpu)

    item2idx, user2idx, idx2item, idx2user = load_index_file()

    item_data['item_id'] = item_data.loc[:, 'url'].map(item2idx)
    item_data.drop('url', axis = 1, inplace= True)

    review_data['item_id'] = review_data.loc[:, 'wine_url'].map(item2idx)
    review_data['user_id'] = review_data.loc[:, 'user_url'].map(user2idx)
    review_data.drop(['user_url','wine_url'], axis = 1, inplace= True)

    ####추가
    feature_engineering()

    item_data.to_csv('/opt/ml/wine/data/item_data.csv', encoding='utf-8-sig', index=False)
    review_data.to_csv('/opt/ml/wine/data/review_data.csv', encoding='utf-8-sig', index=False)

    user_data = review_data.groupby('user_id').agg(count=('rating', 'count'), mean=('rating', 'mean')).reset_index()


    print(f'Total {len(item_data)} items, {len(user_data)} users, {len(review_data)} interactions')

    review_data.rename(columns={'rating': 'user_rating','date': 'timestamp'}, inplace=True)


    train_rating = pd.merge(review_data.loc[:,['user_id','user_rating','timestamp','item_id']],
                            item_data.loc[:, 'item_id'],
                            on = 'item_id', how = 'inner')

    train_rating.to_csv('/opt/ml/wine/data/train_rating.csv', encoding='utf-8-sig', index=False)
    user_data.to_csv('/opt/ml/wine/data/user_data.csv', encoding='utf-8-sig', index=False)


    return train_rating, user_data, item_data

def load_data_file():
    data_path = '/opt/ml/wine/data'
    # train load
    try:
        train_data = pd.read_csv(os.path.join(data_path, 'train_rating.csv'))
        user_data = pd.read_csv(os.path.join(data_path, 'user_data.csv'))
        item_data = pd.read_csv(os.path.join(data_path, 'item_data.csv'))
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
    train_data.columns = to_recbole_columns(train_data.columns)
    user_data.columns = to_recbole_columns(user_data.columns)
    item_data.columns = to_recbole_columns(item_data.columns)

    # to_csv
    outpath = f"/opt/ml/wine/dataset/{dataset_name}"
    os.makedirs(outpath, exist_ok=True)

    train_data.to_csv(os.path.join(outpath,"train_data.inter"),sep='\t',index=False, encoding='utf-8-sig')
    item_data.to_csv(os.path.join(outpath,"train_data.item"),sep='\t',index=False, encoding='utf-8-sig')
    user_data.to_csv(os.path.join(outpath,"train_data.user"),sep='\t',index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    prepare_dataset()
    prepare_recbole_dataset()