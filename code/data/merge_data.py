import pandas as pd
import numpy as np
import joblib
import ast
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import joblib
import ast
from joblib import Parallel, delayed
import json
import os
from tqdm import tqdm
import os, pdb
from datetime import datetime
import pickle
import argparse


def merge(args):
    for i in tqdm(range(6)):

        if i == 0:
            data = pd.read_csv(f'/opt/ml/wine/data/{args.dataset_name}_total.csv', encoding = 'utf-8-sig')
            data.dropna(inplace=True)
        else:
            tmp = pd.read_csv(f'/opt/ml/wine/data/{args.dataset_name}{i}.csv', encoding = 'utf-8-sig')
            tmp.dropna(inplace=True)
            data = pd.concat([data, tmp], axis=0)

    tmp = pd.read_csv(f'/opt/ml/wine/data/{args.dataset_name}.csv', encoding = 'utf-8-sig')
    data = pd.concat([data, tmp], axis=0)

    bf = len(data)
    data.drop_duplicates(inplace = True)
    
    af = len(data)
    data.to_csv(f'/opt/ml/wine/data/review_df_total.csv', encoding = 'utf-8-sig', index= False)
    print(f"Before drop duplicates {bf}, After {af}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='review_df', type=str)
    args = parser.parse_args()
    merge(args)