from typing import Union, Tuple, List
from transformers import BertConfig, BertForPreTraining, BertTokenizerFast
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
import matplotlib.pyplot as plt
from transformers import DataCollatorForLanguageModeling
import os
from transformers import Trainer, TrainingArguments
from transformers.utils import logging
logger = logging.get_logger(__name__)
from filelock import FileLock
import time
import unicodedata
import pickle
import json
from summarizer import Summarizer,TransformerSummarizer
from data_utils import parallel_dataframe_1input

def is_english(text):
    # Remove all punctuation and whitespace
    text = ''.join(char for char in text if unicodedata.category(char) != 'Zs' and unicodedata.category(char) != 'P')
    
    # Calculate the percentage of non-English characters
    english_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')
    non_english_chars = len(text) - english_chars
    non_english_percentage = (non_english_chars / len(text)) * 100
    
    # Return True if the text is at least 90% English characters
    return non_english_percentage <= 15

def remove_non_eng(df):
    df['eng_or_na'] = df['text'].progress_apply(is_english)
    eng_review= df[df['eng_or_na']==True]
    return eng_review

with open('/opt/ml/wine/code/feature_map/item2idx.json','r') as f: item2idx = json.load(f)
basic_info = pd.read_csv('/opt/ml/wine/data/wine_df.csv')
basic_info['wine_id'] = basic_info['url'].map(item2idx)
basic_info = basic_info[basic_info['wine_id'].isnull() == False]
review_df = pd.read_csv('/opt/ml/wine/data/review_df_cleaned.csv',encoding = 'utf-8-sig')

basic_info['wine_id'] = basic_info['wine_id'].astype('int').astype('category')
review_df['wine_id'] = review_df['wine_id'].astype('int').astype('category')

tqdm.pandas()


GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")

eng_review = parallel_dataframe_1input(remove_non_eng, review_df, 8)
eng_review = pd.merge(eng_review, basic_info.loc[:,['wine_id','wine_style']], on = 'wine_id',how ='inner')
eng_review =eng_review[eng_review['wine_style'].isna()==False]

merged_reviews = eng_review.groupby('wine_style').agg({'text': ' '.join})
top_styles = basic_info['wine_style'].value_counts().index[:150]
top_reviews = merged_reviews.loc[merged_reviews.index.isin(top_styles)]

style_summary = {}
GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
for style, text in tqdm(zip(top_reviews.index, top_reviews['text'])):
    text = text[:1000000]
    style_summary[style] = ''.join(GPT2_model(text, min_length=30, max_length=100))

with open('/opt/ml/wine/data/wine_style_summary.json','w') as f: json.dump(style_summary, f)