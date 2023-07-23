from typing import Union, Tuple, List
import gc
import numpy as np
import random
import pandas as pd
from datetime import datetime, date
from tqdm.notebook import tqdm
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score
from tqdm import tqdm
# from IPython.display import Image
from joblib import Parallel, delayed
import torch
import argparse
from model import BERTClass
import torch.nn as nn
from torch.nn.init import normal_
from torch.utils.data import TensorDataset, DataLoader
import torch
import pdb
from transformers import BertTokenizer, BertModel, BertTokenizerFast
import re
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
from collections import defaultdict
#logging.basicConfig(level=logging.INFO)
from torch.utils.data.dataset import Dataset
from transformers import DataCollatorForLanguageModeling
import os
from transformers import Trainer, TrainingArguments
from transformers.utils import logging
logger = logging.get_logger(__name__)
from filelock import FileLock
import time
import pickle
import json
from dataset import MultilabelDataset
from transformers import BertForMaskedLM, pipeline
from torch import cuda
import csv
import multiprocessing as mp
import ast
from transformers import Trainer, TrainingArguments, AutoTokenizer
from multiprocessing import Manager, Pool

def get_label_num():
    #model = BERTClass(num_labels= )
    labeled_review = pd.read_csv('/opt/ml/wine/data/sample_labeled_review.csv', 
                        encoding = 'utf-8', 
                        usecols=['wine_id','text','price_label','note_label'],
                        nrows=3)

    labeled_review['note_label'] = labeled_review['note_label'].apply(list2array)
    labeled_review['price_label'] = labeled_review['price_label'].apply(list2array)
    labeled_review['label'] = labeled_review['note_label'].copy()

    for i in tqdm(range(labeled_review.shape[0])):

        labeled_review['label'][i] = np.concatenate((labeled_review['note_label'][i],labeled_review['price_label'][i]))
        
    labeled_review.drop(['note_label','price_label'], axis = 1, inplace = True)
    columns_to_load = ['wine_id','grape_label','winetype_label','country_label']
    wine_label = pd.read_csv('/opt/ml/wine/data/sample_wine_label.csv', 
                                encoding = 'utf-8',
                                usecols=columns_to_load)
    
    wine_label['grape_label'] = wine_label['grape_label'].apply(list2array)
    wine_label['winetype_label'] = wine_label['winetype_label'].apply(list2array)
    wine_label['country_label'] = wine_label['country_label'].apply(list2array)
    wine_label['label'] = wine_label['grape_label'].view()

    for i in tqdm(wine_label.index):
        wine_label['label'][i] =np.concatenate((wine_label['grape_label'][i],
                                                wine_label['winetype_label'][i],
                                                wine_label['country_label'][i]))

    wine_label.drop(['grape_label','winetype_label','country_label'], axis = 1, inplace = True)
    num = len(labeled_review['label'][0]) + len(wine_label['label'].iloc[0])
    del wine_label
    del labeled_review
    gc.collect()
    return num


def list2array(x):
    return np.array(ast.literal_eval(x), dtype=np.int8)

def get_embedding_multilabel(df, vector_dic, args):

    device = torch.device("cuda")
    df.reset_index(inplace = True)
    review_vectors = {}


    model = BERTClass(
        num_labels= get_label_num(),
        mode = 'embedding'
        )
    

    model.load_state_dict(torch.load(args.model_path))
    model.to(device)


    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    params = {'batch_size': args.batch_size,
                'shuffle': False,
                'num_workers': 0
                }
    
    dataset = MultilabelDataset('inference',df, None, tokenizer, args.max_len)
    data_loader = DataLoader(dataset, **params)
    vector_dic = defaultdict(list)
    with torch.no_grad():
        for data in tqdm(data_loader):
            ids = data['ids'].to(args.device, dtype = torch.long)
            mask = data['mask'].to(args.device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(args.device, dtype = torch.long)
            wine_ids = data['wine_ids'].tolist()
            outputs = model(ids, mask, token_type_ids)
            for id, vector in zip(wine_ids, outputs):
                if str(id) not in vector_dic:
                    vector_dic[str(id)] = {'count': 1, 'mean': vector}
                else:
                    vector_dic[str(id)]['count'] += 1
                    vector_dic[str(id)]['mean'] += (vector - vector_dic[str(id)]['mean']) / vector_dic[str(id)]['count']
                del vector
                del id
                gc.collect()
                
            del outputs
            del ids, mask, token_type_ids, wine_ids
            gc.collect()
    return vector_dic


def get_embedding_MLM(df, args):

    device = torch.device("cuda")
    df.reset_index(inplace = True)
    review_vectors = {}

    tokenizer = BertTokenizerFast(
                vocab_file='/opt/ml/wine/code//text/models/review_tokenizer-vocab.txt',
                max_len=156,
                do_lower_case=True
                )
            
    model = BertModel.from_pretrained('/opt/ml/wine/code/text/models/model_output')
    model.to(device)

    with torch.no_grad():

        for i in tqdm(range(len(df))):
            reviews = df['text'][i].split('.')[:500]
            id = str(df['wine_id'][i])
            review_vector = []
            for text in tqdm(reviews):
                try:
                    encoded_input = tokenizer.encode_plus(
                        text, 
                        truncation = True,
                        add_special_tokens=True, 
                        return_tensors='pt')
                    for key in encoded_input:
                        encoded_input[key] = encoded_input[key].to(device)

                    model_output = model(**encoded_input)
                
                    embeddings = model_output.last_hidden_state.detach().cpu()
                    sentence_embedding = torch.mean(embeddings[0], dim=0)
                    review_vector.append(sentence_embedding)
                    del embeddings
                    del model_output
                    del encoded_input
                    gc.collect()
                except: 1
            mean_vector = torch.mean(torch.stack(review_vector), dim=0).numpy()
            del review_vector
            del sentence_embedding
            gc.collect()
            review_vectors[id] = mean_vector
            
    return review_vectors

def parallel_embedding(df, args, num_cpu):


    chunks = np.array_split(df, num_cpu)
    if args.mode == 'total':
        manager = Manager()
        vector_dic = manager.dict()
        print('Parallelizing with ' + str(num_cpu)+ 'cores')
        with Parallel(n_jobs = num_cpu, backend="multiprocessing") as parallel:
            results = parallel(delayed(get_embedding_multilabel)(chunks[i], vector_dic, args) for i in range(num_cpu))
        pdb.set_trace()

    else:
        print('Parallelizing with ' + str(num_cpu)+ 'cores')
        with Parallel(n_jobs = num_cpu, backend="multiprocessing") as parallel:
            results = parallel(delayed(get_embedding_MLM)(chunks[i], args) for i in range(num_cpu))

        for i,data in enumerate(results):
            if i == 0:
                output = data
            else:
                output.update(data)

    return output


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='total', type=str)
    parser.add_argument("--model_path", 
                        default='/opt/ml/wine/code/text/models/model_outputmodel_state_dict_4.pt', 
                        type=str)
    parser.add_argument("--data", default='/opt/ml/wine/data/review_df_cleaned.csv', type=str)
    parser.add_argument("--max_len", default = 152, type=int)
    parser.add_argument("--batch_size", default = 32, type=int)
    parser.add_argument("--device", default = 'cuda' if cuda.is_available() else 'cpu', type=str)
    print('cuda' if cuda.is_available() else 'cpu')
#######Data#############################################################################
    
    
    
    #mp.set_start_method('spawn')
    torch.multiprocessing.set_start_method('spawn')

    args = parser.parse_args()
    data = pd.read_csv(args.data, encoding= 'utf-8-sig')
    with open('/opt/ml/wine/code/data/feature_map/item2idx.json','r') as f:
        item2idx = json.load(f)
    if 'wine_url' in data.columns:
        data['wine_id'] = data['wine_url'].map(item2idx)
        data = data[data['wine_id'].isna()==False]
        data.reset_index(drop=True, inplace=True)
    review_vectors = parallel_embedding(data, args, 8)
    #get_embedding(data, args)

    try:
        for key in tqdm(review_vectors): review_vectors[key] = review_vectors[key].tolist()
        with open('/opt/ml/wine/data/wine_vector_multilabel.json', 'w') as f: json.dump(review_vectors, f)    
    except:
        import pdb
        pdb.set_trace()
    
    