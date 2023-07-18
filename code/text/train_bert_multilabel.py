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
import pdb
pd.set_option('mode.chained_assignment', None)

def list2array(x):
    return np.array(ast.literal_eval(x), dtype=np.int8)

def convert_label(df, label):
    df[label] = df[label].apply(list2array)
    return df


def run(args, model, optimizer,training_loader,testing_loader):
    best_accuracy = 0
    for epoch in range(args.epochs):
        model = train(args, model, optimizer, training_loader, epoch)
        outputs, targets = validation(args, model, epoch,testing_loader)

        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score : {accuracy}, F1 Score (Micro) : {f1_score_micro}, F1 Score (Macro) : {f1_score_macro}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(args.model_out_path + 'model_state_dict_{epoch}.pt') )
    
    return model


def main(args):
    #tqdm.pandas()
    if args.mode == 'total':
        print("################Preprocessing labeled_review.csv##############")
        labeled_review = pd.read_csv('/opt/ml/winery/data/labeled_review.csv', 
                           encoding = 'utf-8', usecols=['wine_id','text','price_label','note_label'])
               
        #import pdb
        #pdb.set_trace()
        print("note_label")
        labeled_review['note_label'] = labeled_review['note_label'].apply(list2array)
        print('price_label')
        labeled_review['price_label'] = labeled_review['price_label'].apply(list2array)
        print('label')
        labeled_review['label'] = labeled_review['note_label'].copy()

        for i in tqdm(range(labeled_review.shape[0])):
    
            labeled_review['label'][i] = np.concatenate((labeled_review['note_label'][i],labeled_review['price_label'][i]))
            
        labeled_review.drop(['note_label','price_label'], axis = 1, inplace = True)
        print('done for preprocessing labeled review')
#####################Wine
        print("################Preprocessing wine_label.csv##############")
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

        for i in tqdm(wine_label.index):
            wine_label['label'][i] =np.concatenate((wine_label['grape_label'][i],
                                                    wine_label['winetype_label'][i],
                                                    wine_label['country_label'][i]))
    #     wine_label['label'] = np.concatenate([wine_label['grape_label'], 
    #                                        wine_label['winetype_label'],
    #                                        wine_label['country_label']])
        wine_label.drop(['grape_label','winetype_label','country_label'], axis = 1, inplace = True)
        
        data = labeled_review
        
    else:
        data = pd.read_csv(args.data, encoding = 'utf-8')
        data['label'] = data['label'].apply(list2array)

    train_size = args.train_size
    train_dataset = data.sample(frac=train_size,random_state=200)
    test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = data.reset_index(drop=True)

    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if args.mode == 'total':
        wine_label = wine_label.set_index('wine_id')
        training_set = MultilabelDataset(train_dataset, wine_label, tokenizer, args.max_len)
        testing_set = MultilabelDataset(test_dataset, wine_label, tokenizer, args.max_len)
    else:
        training_set = MultilabelDataset(train_dataset, None, tokenizer, args.max_len)
        testing_set = MultilabelDataset(test_dataset, None, tokenizer, args.max_len)   
    # Defining some key variables that will be used later on in the training


    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    if args.mode == 'total':
        pdb.set_trace()
        model = BERTClass(num_labels= len(data['label'][0]) + len(wine_label['label'].iloc[0]))
        
    else:
        model = BERTClass(num_labels= len(data['label'][0]))

    model.to(args.device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.lr)

    model = run(args, model, optimizer, training_loader, testing_loader)
    
    torch.save(model.state_dict(), os.path.join(args.model_output_path + 'model_state_dict.pt') )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='total', type=str)
    parser.add_argument("--model_output_path", default='/opt/ml/winery/code/text/models/model_output', type=str)
    parser.add_argument("--data", default='/opt/ml/winery/data/text_with_notelabel.csv', type=str)
    parser.add_argument("--device", default = 'cuda' if cuda.is_available() else 'cpu', type=str)
#######Data#############################################################################
    parser.add_argument("--max_len", default=156, type=int)
    parser.add_argument("--train_size", default=0.8, type=float)

#######Model#############################################################################
    parser.add_argument("--n_layers", default=8, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--n_heads", default=8, type=int)
    parser.add_argument("--inter_size", default=3072, type=int)
    parser.add_argument("--max_pos_emb", default=128, type=int)

    parser.add_argument("--hddn_d_prob", default=0.1, type=float)
    parser.add_argument("--attn_d_prob", default=0.1, type=float)
    
#######Train#############################################################################
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=1e-05, type=float)
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument("--save_total_limit", default=2, type=int)
     
    args = parser.parse_args()
    
    main(args)
    if not os.path.exists(args.model_out_path):
        os.makedirs(args.model_out_path)
    