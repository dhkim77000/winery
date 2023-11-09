import os
import json
import argparse
import pandas as pd
import numpy as np
import time, datetime
from tqdm import tqdm
from args import parse_args
from logging import getLogger
import torch
from recbole.quick_start import run_recbole

from recbole.config import Config
from recbole.data import dataset
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color

data_path = os.getcwd()
def run(args):
    if args.mode == 'cont':
        return run_recbole(
            model=args.model_name,
            dataset='train_data',
            config_file_list=['/opt/ml/wine/Recbole/basic.yaml'],
        )
    elif args.mode == 'binary':
        return run_recbole(
            model=args.model_name,
            dataset='train_data_binary',
            config_file_list=['/opt/ml/wine/Recbole/binary.yaml'],
        )
    

    files = os.listdir('/opt/ml/wine/Recbole/saved')
    files = [file for file in files if os.path.isfile(os.path.join('/opt/ml/wine/Recbole/saved', file))]
    
    sorted_files = sorted(files, key=lambda x: os.path.getmtime(os.path.join('/opt/ml/wine/Recbole/saved', x)), reverse=True)

    most_recent_model = sorted_files[0]

    os.rename(most_recent_model, os.path.join('/opt/ml/wine/Recbole/saved', 'DCN-latest.pth'))



def main(args):
    # 메모리 부족 문제 해결을 위해 CUDA 캐시 비우기
    torch.cuda.empty_cache()
    # run
    print(f"running {args.model_name}...")
    start = time.time()
    result = run(args)
    t = time.time() - start
    print(f"It took {t/60:.2f} mins")
    print(result)
    
    #wandb.run.finish()
if __name__ == "__main__":
    args = parse_args()
    main(args)