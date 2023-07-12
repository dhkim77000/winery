from typing import Union, Tuple, List
from torch import cuda
import numpy as np
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

from dataset import MultilabelDataset

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 

class BERTClass(torch.nn.Module):
    def __init__(self, num_labels):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, num_labels)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def train(args):
    model.train()
    for data in tqdm(training_loader, 0):
        ids = data['ids'].to(args.device, dtype = torch.long)
        mask = data['mask'].to(args.device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(args.device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del token_type_ids, ids, mask, outputs, loss, data
        gc.collect()

def validation(args, epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(args.device, dtype = torch.long)
            mask = data['mask'].to(args.device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(args.device, dtype = torch.long)
            targets = data['targets'].to(args.device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

            del targets, ids, mask, outputs, data
            gc.collect()

    return fin_outputs, fin_targets

def run(args):
    for epoch in range(args.epochs):
        train(args, model, epoch)
        outputs, targets = validation(args, model, epoch)

        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score : {accuracy}, F1 Score (Micro) : {f1_score_micro}, F1 Score (Macro) : {f1_score_macro}")
    
    return 


def main(args):

    data = pd.read_csv(args.data, encoding = 'utf-8')
    try: data['label'] = data['label'].apply(ast.literal_eval)
    except: pass

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

    model = run(args)
    
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
    parser.add_argument("--n_epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-05, type=float)
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument("--save_total_limit", default=2, type=int)
     
    args = parser.parse_args()
    main(args)
