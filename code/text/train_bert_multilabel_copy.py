from typing import Union, Tuple, List
from torch import cuda
import numpy as np
import ast
import random
import transformers
import pandas as pd
from datetime import datetime, date
from tqdm.notebook import tqdm
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score
from tqdm import tqdm
# from IPython.display import Image
from joblib import Parallel, delayed
import torch
import torch.nn as nn
from torch.nn.init import normal_
from torch.utils.data import TensorDataset, DataLoader
import torch
from transformers import BertTokenizer, BertModel
import re
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)
from torch.utils.data.dataset import Dataset
from transformers import DataCollatorForLanguageModeling
import os
from sklearn import metrics
import gc
from transformers import Trainer, TrainingArguments, AutoTokenizer
from transformers.utils import logging
from transformers import BertConfig, BertForPreTraining, BertTokenizerFast
from filelock import FileLock
import time
import pickle
from dataset import TextDatasetForNextSentencePrediction
from transformers import BertConfig, BertForPreTraining
from tokenizers import BertWordPieceTokenizer
import argparse
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score
from model import BERTClass
from dataset import MultilabelDataset
from train_utils import train, validation
# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 


def list2array(x):
    return np.array(ast.literal_eval(x), dtype=np.int8)

    
labeled_review = pd.read_csv('/opt/ml/winery/data/labeled_review.csv', 
                    encoding = 'utf-8', usecols=['wine_id','text','price_label','note_label'])
# print(labeled_review['note_label'][3])
# print(labeled_review['price_label'][3])
# import pdb
# #pdb.set_trace()

labeled_review['note_label'] = labeled_review['note_label'].apply(list2array)
labeled_review['price_label'] = labeled_review['price_label'].apply(list2array)

labeled_review['label'] = labeled_review['note_label'].view()
for i in tqdm(range(3)):
    
    labeled_review['label'][i] =np.concatenate((labeled_review['note_label'][i],labeled_review['price_label'][i]))

print("labeled_review['price_label'][0]:",labeled_review['price_label'][0])
print("labeled_review['note_label'][0]:",labeled_review['note_label'][0])
print("labeled_review['label'][0]:",labeled_review['label'][0])

labeled_review.drop(['note_label','price_label'], axis = 1, inplace = True)
##############
columns_to_load = ['wine_id','grape_label','winetype_label','country_label']
wine_label = pd.read_csv('/opt/ml/winery/data/wine_label.csv', 
                            encoding = 'utf-8',
                            usecols=columns_to_load)
print("grape_label")
wine_label['grape_label'] = wine_label['grape_label'].apply(list2array)
print("winetype_label")
wine_label['winetype_label'] = wine_label['winetype_label'].apply(list2array)
print("country_label")
wine_label['country_label'] = wine_label['country_label'].apply(list2array)

wine_label['label'] = wine_label['grape_label'].view()

for i in tqdm(range(1)):
            wine_label['label'][i] =np.concatenate((wine_label['grape_label'][i],wine_label['winetype_label'][i],wine_label['country_label'][i]))
            

print("wine_label['grape_label'][0]:",wine_label['grape_label'][0])
print("wine_label['winetype_label'][0]:",wine_label['winetype_label'][0])
print("wine_label['country_label'][0]:",wine_label['country_label'][0])
print("wine_label['label'][0]:",wine_label['label'][0])

wine_label.drop(['grape_label','winetype_label','country_label'], axis = 1, inplace = True)
