
import pandas as pd
import faiss
import pdb
from cluster_nns import get_nns
import numpy as np

def string2array(x):
    x = x.replace('\n', '').strip('[]')
    x_list = [float(i) for i in x.split(' ') if len(i) != 0]
    return np.array(x_list)

def candid2recbole(item_data : pd.DataFrame,
                   item_data_recbole : pd.DataFrame,
                   inter : pd.DataFrame):
    
    pdb.set_trace()
    item_data_recbole.set_index('wine_id:token', inplace= True)
    item_data_recbole['wine_id:token'] = item_data_recbole.index


    inter_per_user = inter.groupby('email:token')['wine_id:token'].agg(list)

    item_data.set_index('wine_id', inplace = True)
    item_data['wine_id'] = item_data.index
    item_data['vectors'] = item_data['vectors'].apply(string2array)

    wine_vectors = []
    for vector in item_data['vectors:float_seq']: wine_vectors.append(vector)
    wine_vectors = np.array(wine_vectors)

    wine_ids = list(item_data.index) #####wine id 
    vector_dimension = wine_vectors.shape[0]

    index = faiss.IndexFlatIP(vector_dimension)
    index = faiss.IndexIDMap2(index)
    index.add_with_ids(wine_vectors, wine_ids)

    wine_id_candidates = get_nns(user = 21421, 
                                inter_per_user = inter_per_user,
                                item_data = item_data,
                                index = index,
                                total_k = 15000)
    wine_id_candidates = list(set([x[0] for x in wine_id_candidates]))

    pdb.set_trace()
    candidates = item_data_recbole.loc[wine_id_candidates, :]
    inter = inter[inter['wine_id:token'].isin(candidates)]