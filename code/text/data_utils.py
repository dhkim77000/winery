from typing import Union, Tuple, List
import numpy as np
import random
import pandas as pd
from typing import Union, Tuple, List
import transformers
import torch
import numpy as np
import random
import pandas as pd
from datetime import datetime, date
from tqdm.notebook import tqdm
from tqdm import tqdm
# from IPython.display import Image
from joblib import Parallel, delayed
import re
from transformers import BertConfig, BertForPreTraining, BertTokenizerFast
from filelock import FileLock
import time
import pickle
from transformers import BertConfig, BertForPreTraining
from tokenizers import BertWordPieceTokenizer
import argparse

from transformers import Trainer, TrainingArguments, AutoTokenizer
from datetime import datetime, date
from tqdm.notebook import tqdm
from tqdm import tqdm
# from IPython.display import Image
from joblib import Parallel, delayed
import json
import ast
from collections import defaultdict

def keep_english_and_digits(text):
    # Remove any characters that are not English alphabets, digits, periods, or commas at the end of sentences
    clean_text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
    return clean_text

def str2list(x):
    try: return ast.literal_eval(x)
    except: return ['None']

def str2dict(x):
    try: return ast.literal_eval(x)
    except: return {}
    
def get_keys(x):
    return set(x.keys())

def get_all_grapes(wine_info):
    
    unique_grape = defaultdict(int)
    for grapes in tqdm(wine_info['grapes']):
        for grape in grapes:
            grape = re.sub(r'\d+%_', '', grape).lower()
            unique_grape[grape] += 1

    return dict(sorted(unique_grape.items(), key=lambda x: x[1], reverse=True))

def grape_encoding(wine_info, grape2idx, threshold):
    wine_info.reset_index(drop = True, inplace= True)
    grape_onehot = []
    for grapes in wine_info['grapes']:
        tmp = [0 for _ in range(len(grape2idx))]
        for grape in grapes:
            if grape in grape2idx:
                matches = re.findall(r'(\d+)%_(\w+)', grape)

                if len(matches) == 1:
                    percentage, grape  = matches[0]
                    if int(percentage) >= threshold: tmp[grape2idx[grape]] = 1

                else: tmp[grape2idx[grape]] = 1
        grape_onehot.append(tmp)
    wine_info['grape_label'] = grape_onehot
    return wine_info

def get_notes_group(wine_df):
    note_df = wine_df.filter(like='_child')

    notes = {}

    for col in note_df.columns:
        note_df.loc[:,col] = note_df.loc[:,col].apply(str2dict)
        sub_note = set()
        for i in tqdm(range(len(note_df))):
            subs = get_keys(note_df[col][i])
            sub_note = sub_note | subs
        notes[col.replace('_child','')] = sub_note

    return notes

def check_note_in_review(text, notes_data):
    text = text.lower()
    result = []
    for key in notes_data:
        if any(word in text for word in notes_data[key]):
            result.append(1)
        else: result.append(0)
    return result

def marking_note_data(df, notes_data):
    df.reset_index(inplace = True)
    data = []
    for i in tqdm(range(len(df))):
        note_onehot = check_note_in_review(df.loc[i,'text'], notes_data)
        tmp = {}
        tmp['wine_id'] = df.loc[i,'wine_id']
        tmp['text'] = df.loc[i,'text']
        tmp['note_label'] = note_onehot
        data.append(tmp)
        
    return pd.DataFrame(data)

def parallel_dataframe_2input(func, df, mapping_data, num_cpu):
    try:
        for key in mapping_data: mapping_data[key] = set(mapping_data[key])
    except: pass

    chunks = np.array_split(df, num_cpu)

    print('Parallelizing with ' +str(num_cpu)+'cores')
    with Parallel(n_jobs = num_cpu, backend="multiprocessing") as parallel:
        results = parallel(delayed(func)(chunks[i], mapping_data) for i in range(num_cpu))

    for i,data in enumerate(results):
        if i == 0:
            output = data
        else:
            output = pd.concat([output, data], axis=0)
    output.reset_index(inplace = True, drop = True)
    return output

def parallel_dataframe_1input_col(func, df, col, num_cpu):
 
    chunks = np.array_split(df, num_cpu)

    print('Parallelizing with ' +str(num_cpu)+'cores')
    with Parallel(n_jobs = num_cpu, backend="multiprocessing") as parallel:
        results = parallel(delayed(func)(chunks[i], col) for i in range(num_cpu))

    for i,data in enumerate(results):
        if i == 0:
            output = data
        else:
            output = pd.concat([output, data], axis=0)
    output.reset_index(inplace = True, drop = True)
    return output

def cut_lowcount_feat(df, col, threshold):
    feats = {}
    idx = 1
    for feature, count in zip(df[col].value_counts().index,  df[col].value_counts()):
        if count >= threshold:
            feats[feature] = idx
            idx += 1
        else:
            feats[feature] = 0
    return feats

def string2label(df, column, feature2idx):
    labels = []
    df[column] = df[column].fillna("None")
    for feature in tqdm(df[column]):
        tmp = [0 for _ in range(max(feature2idx.values())+1)]

        if feature in feature2idx:
            tmp[feature2idx[feature]] = 1
        else: tmp[-1] = 0

        labels.append(tmp)
        
    df[f"{column}_label"] = labels
    return df

def gen_labeled_data(
        wine_info : pd.DataFrame, 
        grape_threshold : int = 89, 
        grape_percentage_threshold : int = 20,
        country_threshold : int = 150,
        winetype_threshold : int = 0
        ):
    wine_info.drop_duplicates(inplace = True)
    wine_info = wine_info.loc[:,['wine_id','country','grapes','winetype']]
    wine_info['country'].fillna('other', inplace = True)
    wine_info['winetype'].fillna('other', inplace = True)
    wine_info['grapes'] = wine_info['grapes'].apply(str2list)

    unique_grape = get_all_grapes(wine_info)
    filtered_grape = {k: v for k, v in unique_grape.items() if v >=grape_threshold}
    grape2idx =  {k: v for v, k in enumerate(filtered_grape.keys())}

    wine_info = grape_encoding(wine_info, grape2idx, grape_percentage_threshold)
    sum_zero_filter = lambda x: sum(x) != 0

    # Apply the filter and drop rows where the sum of the list is 0
    wine_info = wine_info[wine_info['grape_label'].apply(sum_zero_filter)]

    country2idx =  cut_lowcount_feat(wine_info, 'country', country_threshold)
    winetype2idx = cut_lowcount_feat(wine_info, 'winetype', winetype_threshold)

    wine_info = string2label(wine_info, 'winetype', winetype2idx)
    wine_info = string2label(wine_info, 'country', country2idx)
    wine_info.reset_index(inplace = True, drop = True)
    return wine_info, grape2idx, country2idx, winetype2idx

def check_price_in_review(text, price_vocab):
    price_vocab = set(price_vocab)
    text = text.lower()
    if any(word in text for word in price_vocab): return [1]
    else: return [0]

def marking_price_data(df, price_vocab):
    df.reset_index(inplace = True, drop = True)
    df['text'] = df['text'].fillna('')
    data = []
    for i in tqdm(range(len(df))):
        tmp = {}
        tmp['wine_id'] = df.loc[i,'wine_id']
        tmp['text'] = df.loc[i,'text']
        tmp['note_label'] = df.loc[i,'note_label']
        tmp['price_label'] = check_price_in_review(df.loc[i,'text'], price_vocab)
        data.append(tmp)
        
    return pd.DataFrame(data)

def get_len_text(x):
    return len(x.split())

def merge_short_review(df, threshold):
    df.reset_index(inplace = True, drop = True)
    data = []

    prv_text = ''
    prv_id = df.loc[0, 'wine_id']
    
    for i in tqdm(range(len(df))):
        tmp = {}
        
        wine_id = df.loc[i, 'wine_id']
        length = df.loc[i, 'length']
        text = df.loc[i, 'text']
        
        if length <= threshold: #######short text
            if (length + len(prv_text.split()) > threshold):
                if wine_id == prv_id:
                    text += prv_text
                    length += len(prv_text.split())

                tmp['wine_id'] = wine_id
                tmp['text'] = text
                tmp['length'] = length 
                data.append(tmp)
                prv_text = text
                prv_id = wine_id
                
            else: prv_text += text
        else: #######long text
            tmp['wine_id'] = wine_id
            tmp['text'] = text
            tmp['length'] = length
            data.append(tmp)
            prv_text = text
            prv_id = wine_id
        
    result = pd.DataFrame(data)
    result = result[result['length'] >= threshold]
    result.drop_duplicates(inplace = True)
    result.dropna(inplace= True)
    result.reset_index(inplace = True, drop = True)
    
    print(f"Before : {len(df)}")
    print(f"After : {len(result)}")
    result['wine_id'] = result['wine_id'].astype('category')
    result['length'] = result['length'].astype(int)
    result['text'] = result['text'].astype(str)
    
    return result