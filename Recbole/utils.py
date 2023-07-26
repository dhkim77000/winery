
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

    wine_id_candidates = get_nns(inter_wine_ids = inter_per_user[2141],
                                item_data = item_data,
                                index = index,
                                total_k = 15000)
    
    wine_id_candidates = list(set([x[0] for x in wine_id_candidates]))


    candidates_item_data = item_data_recbole.loc[wine_id_candidates, :]
    candidates_inter = inter[inter['wine_id:token'].isin(wine_id_candidates)]
    candidates_item_emb = item_emb[item_emb['wid:token'].isin(wine_id_candidates)]

    candidates_item_emb.to_csv(os.path.join(outpath,"train_data.itememb"),sep='\t',index=False, encoding='utf-8')
    candidates_inter.to_csv(os.path.join(outpath,"train_data.inter"),sep='\t',index=False, encoding='utf-8')
    candidates_item_data.to_csv(os.path.join(outpath,"train_data.item"),sep='\t',index=False, encoding='utf-8')

    return 


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