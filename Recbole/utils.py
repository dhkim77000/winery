
import pandas as pd
import faiss
import os
import pdb
from cluster_nns import get_nns
import numpy as np
from google.cloud import storage
from datetime import datetime

def data2bucket():
    current_time = datetime.now()
    bucket_name = 'rank_info_db2model'    
    source_file_name = '/home/dhkim/winery/output/inference.json'
    destination_blob_name = f'{current_time}_inference.json'
  
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print("upload success")

def string2array(x):
    x = x.replace('\n', '').strip('[]')
    x_list = [float(i) for i in x.split(' ') if len(i) != 0]
    return np.array(x_list)
                
def candid2recbole(item_data : pd.DataFrame,
                   user_count : int,
                   inter_per_user : pd.DataFrame,
                   popular : set,
                   index : faiss.IndexIDMap2):
    
    #outpath = '/home/dhkim/winery/dataset/cadidates'

    #item_data_recbole.set_index('wine_id:token', inplace= True)
    #item_data_recbole['wine_id:token'] = item_data_recbole.index

    wine_id_candidates = get_nns(inter_wine_ids = inter_per_user[7],
                                item_data = item_data,
                                index = index,
                                total_k = 15000)
    
    wine_id_candidates = set([x[0] for x in wine_id_candidates])
    wine_id_candidates.union(popular)
    wine_id_candidates = list(rule_based(item_data, wine_id_candidates, user_count))


    #candidates_item_data = item_data_recbole.loc[wine_id_candidates, :]
    #candidates_inter = inter[inter['wine_id:token'].isin(wine_id_candidates)]
    #candidates_item_emb = item_emb[item_emb['wid:token'].isin(wine_id_candidates)]

    #candidates_item_emb.to_csv(os.path.join(outpath,"train_data.itememb"),sep='\t',index=False, encoding='utf-8')
    #candidates_inter.to_csv(os.path.join(outpath,"train_data.inter"),sep='\t',index=False, encoding='utf-8')
    #candidates_item_data.to_csv(os.path.join(outpath,"train_data.item"),sep='\t',index=False, encoding='utf-8')

    return wine_id_candidates


def bayesian_average(df, min_votes=20, prior_rating=None):
    v = df['num_votes']
    R = df['wine_rating']
    
    if prior_rating is None:
        C = df['wine_rating'].mean()  # Mean rating as the prior rating
    else:
        C = prior_rating

    m = min_votes

    df['popularity_adjusted_rating'] = ((v / (v + m)) * R) + ((m / (v + m)) * C)
    return df


def rule_based(item_data, candidates, user_count):

################################# HEAVY ################################# 
    if user_count >= 50: 
        expensive_wine =  set(item_data[item_data['price'] > 75]['wine_id'])
        expensive_dessert_wine = set(item_data[(item_data['price'] > 100) & (item_data['winetype'] == 'dessertwine')]['wine_id'])
        except_cheap_rose_wine = set(item_data[(item_data['price'] > 50) & (item_data['winetype'] == 'Rose')]['wine_id'])

        
        candidates.union(expensive_dessert_wine)
        candidates.union(expensive_wine)
        candidates.difference(except_cheap_rose_wine)

################################# MEDIUM ################################# 
    elif (user_count >=5) & (user_count < 50) :
        dessert_wine = set(item_data[item_data['winetype'] == 'dessertwine']['wine_id'])
        Acidic_wine_90 = set(item_data[item_data['Acidic']>=90]['wine_id'])
        normal_price_wine =  set(item_data[(item_data['price'] >= 50) & (item_data['price'] <= 100)]['wine_id'])

        candidates.union(dessert_wine)
        candidates.union(normal_price_wine)
        candidates.difference(Acidic_wine_90)
        

################################# LIGHT ################################# 
    elif user_count < 5:
        dessert_wine = set(item_data[item_data['winetype'] == 'dessertwine']['wine_id'])
        expensive_dessert_wine = set(item_data[(item_data['price'] > 100) & (item_data['winetype'] == 'dessertwine')]['wine_id'])

        sweet_wine = set(item_data[item_data['Sweet'] > 7]['wine_id'])
        Acidic_wine_80 = set(item_data[item_data['Acidic']>=80]['wine_id'])

        candidates.union(expensive_dessert_wine)
        candidates.union(dessert_wine)
        candidates.union(sweet_wine)
        candidates.difference(Acidic_wine_80)

    return candidates