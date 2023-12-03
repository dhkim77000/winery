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
from transformers import BertConfig, BertForPreTraining, BertTokenizerFast
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
    labeled_review = pd.read_csv('/home/dhkim/server_front/winery_AI/winery/data/sample_labeled_review.csv', 
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
    wine_label = pd.read_csv('/home/dhkim/server_front/winery_AI/winery/data/sample_wine_label.csv', 
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

    if args.mode == 'total':
        model = BERTClass(
            args = args,
            num_labels= get_label_num(),
            mode = 'embedding'
            )
    else:
        model = BERTClass(
            args = args,
            num_labels= len([0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]),
            mode = 'embedding'
            )
        
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)


    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    params = {'batch_size': args.batch_size,
                'shuffle': False,
                'num_workers': 0
                }
    
    dataset = MultilabelDataset('inference_no_label', df, None, tokenizer, args.max_len)
    data_loader = DataLoader(dataset, **params)

    answer_ids = df['index']
    i = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            ids = data['ids'].to(args.device, dtype = torch.long)
            mask = data['mask'].to(args.device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(args.device, dtype = torch.long)
            output = model(ids, mask, token_type_ids)
            vector_dic[answer_ids[i]] = output.tolist()
            i+=1

    with open('/home/dhkim/server_front/winery_AI/winery/data/mbti_vector.json','w') as f: json.dump(vector_dic, f)

    return vector_dic



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='total', type=str)
    parser.add_argument("--model_path", 
                        default='/home/dhkim/server_front/winery_AI/winery/code/text/models/model_outputmodel_state_dict_4.pt', 
                        type=str)
    parser.add_argument("--max_len", default = 152, type=int)
    parser.add_argument("--batch_size", default = 1, type=int)
    parser.add_argument("--device", default = 'cuda' if cuda.is_available() else 'cpu', type=str)
    parser.add_argument("--pool", default = 'mean', type=str)
    print('cuda' if cuda.is_available() else 'cpu')
#######Data#############################################################################
    args = parser.parse_args()
 
    with open('/home/dhkim/server_front/winery_AI/winery/data/mbti_answer.json','r') as f:
        mbti_vec = json.load(f)

    
    data = pd.DataFrame(mbti_vec.values(), index= mbti_vec.keys(), columns = ['text'])
    with open('/home/dhkim/server_front/winery_AI/winery/code/data/feature_map/item2idx.json','r') as f:
        item2idx = json.load(f)

    get_embedding_multilabel(data, {}, args)

    

