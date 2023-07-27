
import pandas as pd
import faiss
import os
import pdb
from cluster_nns import get_nns
import numpy as np

def string2array(x):
    x = x.replace('\n', '').strip('[]')
    x_list = [float(i) for i in x.split(' ') if len(i) != 0]
    return np.array(x_list)

def candid2recbole(item_data : pd.DataFrame,
                   item_data_recbole : pd.DataFrame,
                   inter_per_user : pd.DataFrame,
                   inter : pd.DataFrame,
                   item_emb : pd.DataFrame,
                   index : faiss.IndexIDMap2):
    
    outpath = '/opt/ml/wine/dataset/cadidates'

    item_data_recbole.set_index('wine_id:token', inplace= True)
    item_data_recbole['wine_id:token'] = item_data_recbole.index

    wine_id_candidates = get_nns(inter_wine_ids = inter_per_user[7],
                                item_data = item_data,
                                index = index,
                                total_k = 15000)
    
    wine_id_candidates = list(set([x[0] for x in wine_id_candidates]))


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


def rule_based(item_data):
    #  가격이 100이상, 디저트 와인 
    dessert_wine = set(item_data[item_data['winetype'] == 'dessertwine']['wine_id'])
    expensive_dessert_wine = set(item_data[(item_data['price'] > 100) & (item_data['winetype'] == 'dessertwine')]['wine_id'])
    #  저가의 로제 와인은 불호하는 경향이 있음 따라서 일정 가격이상의 와인 추출
    except_cheap_rose_wine = set(item_data[(item_data['price'] > 50) & (item_data['winetype'] == 'Rose')]['wine_id'])
    #  전체와인의 평균이상의 가격(75)을 가진 와인 추천 
    expensive_wine =  set(item_data[item_data['price'] > 75]['wine_id'])
    # 
    normal_price_wine =  set(item_data[(item_data['price'] >= 50) & (item_data['price'] <= 100)]['wine_id'])
    sweet_wine = set(item_data[item_data['Sweet'] > 7]['wine_id'])

    #  신맛의 와인
    Acidic_wine = set(item_data[item_data['Acidic']!=0]['wine_id'])
    # 꽤 신와인부터
    Acidic_wine_60 = set(item_data[item_data['Acidic']>=60]['wine_id'])
    # 많이 신와인부터
    Acidic_wine_80 = set(item_data[item_data['Acidic']>=80]['wine_id'])