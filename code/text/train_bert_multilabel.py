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

    data = pd.read_csv(args.data, encoding = 'utf-8')
    tqdm.pandas()
    try: data['label'] = data['label'].progress_apply(ast.literal_eval)
    except: 
        import pdb
        pdb.set_trace()

    train_size = args.train_size
    train_dataset = data.sample(frac=train_size,random_state=200)
    test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = data.reset_index(drop=True)

    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    training_set = MultilabelDataset(train_dataset, tokenizer, args.max_len)
    testing_set = MultilabelDataset(test_dataset, tokenizer, args.max_len)   
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

    
    model = BERTClass(len(data['label'][0]))
    model.to(args.device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.lr)

    model = run(args, model, optimizer, training_loader, testing_loader)
    
    torch.save(model.state_dict(), os.path.join(args.model_out_path + 'model_state_dict.pt') )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_path", default='/opt/ml/wine/code/text/models/model_output', type=str)
    parser.add_argument("--data", default='data/text_with_notelabel.csv', type=str)
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
