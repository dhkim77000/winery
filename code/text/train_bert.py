from typing import Union, Tuple, List

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
from sklearn import metrics


def main(args):
    logger = logging.get_logger(__name__)

    if args.tokenizer == 'auto':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif args.tokenizer =='wp':
        tokenizer = BertTokenizerFast(
            vocab_file=args.vocab_file,
            max_len=args.max_len,
            do_lower_case=True,
            )

    config = BertConfig(    
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layers,    # layer num
        num_attention_heads=args.n_heads,    # transformer attention head number
        intermediate_size=args.inter_size,   # transformer 내에 있는 feed-forward network의 dimension size
        hidden_act="gelu",
        hidden_dropout_prob=args.hddn_d_prob,
        attention_probs_dropout_prob=args.attn_d_prob,
        max_position_embeddings=args.max_pos_emb,    # embedding size 최대 몇 token까지 input으로 사용할 것인지 지정
        pad_token_id=0,
        position_embedding_type="absolute"
    )

    training_args = TrainingArguments(
        output_dir=args.trainer_output_path,
        overwrite_output_dir=True,
        num_train_epochs=args.n_epochs,
        per_gpu_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps
    )
    model = BertForPreTraining(config=config)

    dataset = TextDatasetForNextSentencePrediction(
        logger = logger,
        tokenizer=tokenizer,
        file_path=args.text_file,
        block_size=args.block_size,
        overwrite_cache=False,
        short_seq_probability=args.short_seq_prob,
        nsp_probability=0.5,
    )

    data_collator = DataCollatorForLanguageModeling(    # [MASK] 를 씌우는 것은 저희가 구현하지 않아도 됩니다! :-)
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    

    trainer.train() # wiki 전체 데이터로 학습 시, 1 epoch에 9시간 정도 소요됩니다!! 
    trainer.save_model(args.output_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_path", default='/opt/ml/wine/code/text/models/model_output', type=str)
    parser.add_argument("--trainer_output_path", default='/opt/ml/wine/code/text/models', type=str)
    parser.add_argument("--vocab_file", default='/opt/ml/wine/code/text/models/review_tokenizer-vocab.txt', type=str)
    parser.add_argument("--text_file", default="/opt/ml/wine/data/text_data.txt", type=str)
#######Data#############################################################################
    parser.add_argument("--max_len", default=128, type=int)
    parser.add_argument("--vocab_size", default=40000, type=int)
    parser.add_argument("--min_frequency", default=3, type=int)
    parser.add_argument("--block_size", default=128, type=int)
    parser.add_argument("--short_seq_prob", default=0.1, type=float)
    parser.add_argument("--tokenizer", default='auto', type=str)

#######Model#############################################################################
    parser.add_argument("--n_layers", default=8, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--n_heads", default=8, type=int)
    parser.add_argument("--inter_size", default=3072, type=int)
    parser.add_argument("--max_pos_emb", default=128, type=int)

    parser.add_argument("--hddn_d_prob", default=0.1, type=float)
    parser.add_argument("--attn_d_prob", default=0.1, type=float)
    
#######Train#############################################################################
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument("--save_total_limit", default=2, type=int)
     
    args = parser.parse_args()
    main(args)