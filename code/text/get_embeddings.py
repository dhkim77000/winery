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
import torch.nn as nn
from torch.nn.init import normal_
from torch.utils.data import TensorDataset, DataLoader
import torch
from transformers import BertTokenizer, BertModel, BertTokenizerFast
import re
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
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
from transformers import BertForMaskedLM, pipeline
from torch import cuda

def get_embedding(df):

    device = torch.device("cuda")
    df.reset_index(inplace = True)
    review_vectors = {}
    tokenizer = BertTokenizerFast(
        vocab_file='/opt/ml/wine/code//text/models/review_tokenizer-vocab.txt',
        max_len=156,
        do_lower_case=True,
    )
    model = BertModel.from_pretrained('/opt/ml/wine/code/text/models/model_output')
    model.to(device)
    with torch.no_grad():

        for i in tqdm(range(len(df))):
            reviews = df['text'][i].split('.')[:500]
            id = df['wine_id'][i]

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


def parallel_embedding(df, num_cpu):

    chunks = np.array_split(df, num_cpu)

    print('Parallelizing with ' + str(num_cpu)+ 'cores')
    with Parallel(n_jobs = num_cpu, backend="multiprocessing") as parallel:
        results = parallel(delayed(get_embedding)(chunks[i]) for i in range(num_cpu))

    for i,data in enumerate(results):
        if i == 0:
            output = data
        else:
            output.update(data)

    return output

if __name__ == "__main__":
    merged_reviews = pd.read_csv('/opt/ml/wine/data/merged_review.csv',encoding='utf-8-sig')
    
    #review_vectors = get_embedding(merged_reviews)
    

    review_vectors = parallel_embedding(merged_reviews, 8)
    
    with open('/opt/ml/wine/data/wine_vector.json', 'w') as f:
        json.dump(review_vectors, f)