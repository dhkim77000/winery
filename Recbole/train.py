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
import shutil

from recbole.config import Config
from recbole.data import dataset
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color
import pdb

data_path = os.getcwd()

def run(args):
    if args.mode == 'cont':
        return run_recbole(
            model=args.model_name,
            dataset='train_data',
            config_file_list=['/home/dhkim/server_front/winery_AI/winery/Recbole/basic.yaml'],
        )
    elif args.mode == 'binary':
        return run_recbole(
            model=args.model_name,
            dataset='train_data_binary',
            config_file_list=['/home/dhkim/server_front/winery_AI/winery/Recbole/binary.yaml'],
        )



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

    saved_folder = '/home/dhkim/server_front/winery_AI/winery/saved'
    files = os.listdir(saved_folder)
    files = [file for file in files if os.path.isfile(os.path.join(saved_folder, file))]
    
    sorted_files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(saved_folder, x)), reverse=True)
    pdb.set_trace()
    most_recent_model = os.path.join(saved_folder, sorted_files[0])

    shutil.copy2(most_recent_model,  os.path.join(saved_folder, 'DCN-latest.pth'))
    os.remove(most_recent_model)


    
    #wandb.run.finish()
if __name__ == "__main__":
    args = parse_args()
    main(args)