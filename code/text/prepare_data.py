from typing import Union, Tuple, List
import numpy as np
import random
import pandas as pd
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
import argparse
import gc
from datetime import datetime, date
from tqdm.notebook import tqdm
from tqdm import tqdm
# from IPython.display import Image
from joblib import Parallel, delayed
import json
import ast
from data_utils import *
import pdb
import nltk

def main(args):
    nltk.data.path.append('/opt/ml/wine/code/nltk_data')
    
    # Now download the punkt tokenizer data
    nltk.download('punkt')
    print('-------------------Reading Datas-------------------')
    with open('/opt/ml/wine/code/data/feature_map/item2idx.json','r') as f: item2idx = json.load(f)
    with open('/opt/ml/wine/code/data/feature_map/price_vocab.json','r') as f: price_vocab = json.load(f)
    with open('/opt/ml/wine/code/data/feature_map/note.json','r') as f: notes_data = json.load(f)

    basic_info = pd.read_csv('/opt/ml/wine/data/basic_info_total.csv')
    wine_df = pd.read_csv('/opt/ml/wine/data/wine_df.csv')
    #########################WINE LABEL#########################  
    wine_label = wine_df.loc[:, ['url','winetype']].merge(basic_info, on='url')
    wine_label['wine_id'] = wine_label['url'].map(item2idx)
    wine_label = wine_label[wine_label['wine_id'].isna()==False]
    wine_label['wine_id'] = wine_label['wine_id'].astype('int').astype('category')
    wine_label = wine_label.loc[:,['wine_id','country','grapes','winetype']]
    wine_label, grape2idx, country2idx, winetype2idx = gen_labeled_data(wine_label)
    wine_label.to_csv(args.save_path+'wine_label.csv', index = False)
    wine_ids = wine_label['wine_id'].unique()
#########################REVIEW DATA#########################
    try:
        review_df = pd.read_csv('/opt/ml/wine/data/review_df_cleaned.csv',encoding = 'utf-8-sig')
        review_df = review_df.sample(frac=0.7, random_state=13)
        review_df['wine_id'] = review_df['wine_id'].astype('int').astype('category')
        if args.run == 0:
            print('-------------------Processing review text-------------------')
            review_df = review_df[review_df['text'].isna()==False]
            review_df['text'] = review_df['text'].progress_apply(lambda x: x + '.' if x[-1] != '.' else x)

            review_df['text'] = review_df['text'].progress_apply(keep_english_and_digits)
            review_df['wine_id'] = review_df['wine_url'].map(item2idx)
            review_df = review_df[review_df['wine_id'].isna()==False]
            review_df['wine_id'] = review_df['wine_id'].astype('int').astype('category')
            review_df = review_df[review_df['wine_id'].isin(wine_ids)]
            review_df['length'] = review_df['text'].progress_apply(get_len_text)
            review_df = review_df.loc[:, ['wine_id','text','length']]

            review_df = review_df.sort_values(['wine_id', 'length'])
            review_df = merge_short_review(review_df, args.min_len)
            review_df.to_csv('/opt/ml/wine/data/review_df_cleaned.csv',index = False)

    except Exception as e:
        print(e)
        print('-------------------Processing review text-------------------')
        review_df = pd.read_csv('/opt/ml/wine/data/review_df_total.csv',encoding = 'utf-8-sig').loc[:,['user_url','rating','text','wine_url']]
        tqdm.pandas() 
        review_df = review_df[review_df['text'].notna()]
        review_df['text'] = review_df['text'].progress_apply(lambda x: x + '.' if x[-1] != '.' else x)
        review_df['text'] = review_df['text'].progress_apply(keep_english_and_digits)
        review_df['wine_id'] = review_df['wine_url'].map(item2idx)
        review_df = review_df[review_df['wine_id'].isna()==False]
        review_df['wine_id'] = review_df['wine_id'].astype('int').astype('category')
        
        review_df = review_df[review_df['wine_id'].isin(wine_ids)]

        review_df = review_df[review_df['text'].notna()]
        review_df['length'] = review_df['text'].progress_apply(get_len_text)
        review_df['en'] = review_df['text'].progress_apply(is_mostly_english)
        review_df = review_df[review_df['en'] == True]
        review_df.drop('en', axis = 1, inplace = True)
        review_df = review_df.loc[:, ['wine_id','text','length']]
        review_df = review_df.sort_values(['wine_id', 'length'])
        #review_df = review_df[review_df['length']>=args.min_len]

        review_df = review_df[review_df['length']>=2]
        review_df = merge_short_review(review_df, args.min_len)
        review_df.to_csv('/opt/ml/wine/data/review_df_cleaned.csv',index = False)
    
    gc.collect()

#########################NOTE LABEL#########################
    notes_data = get_notes_group(wine_df)
    
    if args.label == 1:
        review_df = parallel_dataframe_2input(marking_note_data, review_df, notes_data, 8)
        gc.collect()

        #########################PRICE LABEL#########################
        labeled_review = parallel_dataframe_2input(marking_price_data, review_df, price_vocab, 8)
        labeled_review.reset_index(inplace =True, drop = True)
        labeled_review.to_csv(args.save_path+'labeled_review.csv', index = False)
        gc.collect()

        gc.collect()
        with open('/opt/ml/wine/code/data/feature_map/grape2idx.json','w') as f: json.dump(grape2idx, f)
        with open('/opt/ml/wine/code/data/feature_map/country2idx.json','w') as f: json.dump(country2idx, f)
        with open('/opt/ml/wine/code/data/feature_map/winetype2idx.json','w') as f: json.dump(winetype2idx, f)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
#######Train#############################################################################
    parser.add_argument("--min_len", default=6, type=int)
    parser.add_argument("--save_path", default="/opt/ml/wine/data/", type=str)
    parser.add_argument("--run", default=1, type=int)
    parser.add_argument("--label", default=1, type=int)
    args = parser.parse_args()
    main(args)