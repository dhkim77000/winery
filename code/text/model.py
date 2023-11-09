
import torch
import transformers
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
import pdb


class BERTClass(torch.nn.Module):
    def __init__(self, args, num_labels, mode: str = 'train'):
        super(BERTClass, self).__init__()
        self.mode = mode
        self.args = args
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, num_labels)
    
    def forward(self, ids, mask, token_type_ids):
        if self.mode == 'train':
            _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids).values()
            output_2 = self.l2(output_1)
            output = self.l3(output_2)
            return output
        
        elif self.mode == 'embedding':
            if self.args.pool == 'cls':
                output = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
                embeddings = output.last_hidden_state.detach().cpu()
                return embeddings[:,0]
            elif self.args.pool =='mean':
                output = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
                embeddings = torch.mean(output.last_hidden_state, dim=1).detach().cpu()
                return embeddings