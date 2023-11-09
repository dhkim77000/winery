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
import pdb


from recbole.model.general_recommender.multivae import MultiVAE
from recbole.quick_start import run_recbole

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, FullSortEvalDataLoader , Interaction
from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color
from recbole.utils.case_study import full_sort_topk


def main(args):
    """모델 inference 파일
    args
        --inference_model SASRec-Jun-16-2023_14-26-45.pth
        (모델경로)로 사용할 모델을 선택합니다.
        --rank_K로 몇개의 추천아이템을 뽑아낼지 선택합니다.
    """
    
    general_model = ['Pop', 'ItemKNN', 'BPR', 'NeuMF', 'ConvNCF', 'DMF', 'FISM', 'NAIS', 'SpectralCF', 'GCMC', 'NGCF', 'LightGCN', 'DGCF', 'LINE', 'MultiVAE', 'MultiDAE', 'MacridVAE', 'CDAE', 'ENMF', 'NNCF', 'RaCT', 'RecVAE', 'EASE', 'SLIMElastic', 'SGL', 'ADMMSLIM', 'NCEPLRec', 'SimpleX', 'NCL']
    sequence_model = ['FPMC', 'GRU4Rec', 'NARM', 'STAMP', 'Caser', 'NextItNet', 'TransRec', 'SASRec', 'BERT4Rec', 'SRGNN', 'GCSAN', 'GRU4RecF', 'SASRecF', 'FDSA', 'S3Rec', 'GRU4RecKG', 'KSR', 'FOSSIL', 'SHAN', 'RepeatNet', 'HGN', 'HRM', 'NPE', 'LightSANs', 'SINE', 'CORE' ]
    context_aware_model = ['LR', 'FM', 'NFM', 'DeepFM', 'xDeepFM', 'AFM', 'FFM', 'FwFM', 'FNN', 'PNN', 'DSSM', 'WideDeep', 'DIN', 'DIEN', 'DCN', 'DCNV2', 'AutoInt', 'XGBOOST', 'LIGHTGBM' ]
    knowledge_based_model = ['CKE', 'CFKG', 'KTUP', 'KGAT', 'KGIN', 'RippleNet', 'MCCLK', 'MKR', 'KGCN', 'KGNNLS']

    K = args.rank_K

    model_path = 'saved/'+args.saved_model
    model_name = model_path[6:-4].split('-')[0]

    print(model_path,model_name)
    if model_name in general_model :
        checkpoint = torch.load(model_path)
        config = checkpoint['config']
        config['dataset'] = 'train_data'
        print("create dataset start!")
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        print("create dataset done!")
        model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
        model.load_state_dict(checkpoint['state_dict'])
        model.load_other_parameter(checkpoint.get('other_parameter'))

        # device 설정
        device = config.final_config_dict['device']

        # user, item id -> token 변환 array
        user_id = config['USER_ID_FIELD']
        item_id = config['ITEM_ID_FIELD']
        user_id2token = dataset.field2id_token[user_id]
        item_id2token = dataset.field2id_token[item_id]

        # user id list
        user_count = len(user_id2token)
        remainder = user_count % 128
        padding_size = 128 - remainder if remainder != 0 else 0
        all_user_list = torch.arange(0, user_count + padding_size).reshape(-1, 128)

        # user, item 길이
        user_len = len(user_id2token)
        item_len = len(item_id2token)

        # user-item sparse matrix
        matrix = dataset.inter_matrix(form='csr')

        # user id, predict item id 저장 변수
        pred_list = None
        user_list = None

        # model 평가모드 전환
        model.eval()

        # progress bar 설정
        tbar = tqdm(all_user_list, desc=set_color(f"Inference", 'pink'))

        for data in tbar:
            # interaction 생성
            interaction = dict()
            interaction = Interaction(interaction)
            interaction[user_id] = data
            interaction = interaction.to(device)

            # user item별 score 예측
            score = model.full_sort_predict(interaction)
            score = score.view(-1, item_len)

            rating_pred = score.cpu().data.numpy().copy()

            user_index = data.numpy()

            idx = matrix[user_index].toarray() > 0

            rating_pred[idx] = -np.inf
            rating_pred[:, 0] = -np.inf
            ind = np.argpartition(rating_pred, -K)[:, -K:]

            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

            batch_pred_list = ind[
                np.arange(len(rating_pred))[:, None], arr_ind_argsort
            ]

            if pred_list is None:
                pred_list = batch_pred_list
                user_list = user_index
            else:
                pred_list = np.append(pred_list, batch_pred_list, axis=0)
                user_list = np.append(
                    user_list, user_index, axis=0
                )

        result = []
        for user, pred in zip(user_list, pred_list):
            for item in pred:
                result.append((int(user_id2token[user]), int(item_id2token[item])))
             #데이터 저장
        sub = pd.DataFrame(result, columns=["user", "item"])
        print(len(sub))
#          # train load
#         train = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")
#         # indexing save
#         uidx2user = {k:v for k,v in enumerate(sorted(set(train.user)))}
#         iidx2item = {k:v for k,v in enumerate(sorted(set(train.item)))}

#         sub.user = sub.user.map(uidx2user)
#         sub.item = sub.item.map(iidx2item)

#         sub = afterprocessing(args,sub,train)
#         # SAVE OUTPUT
#         output_dir = os.getcwd()+'/output/'
#         write_path = os.path.join(output_dir, f"{model_name}.csv")
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         with open(write_path, 'w', encoding='utf8') as w:
#             print("writing prediction : {}".format(write_path))
#             w.write("user,item\n")
#             for id, p in sub.values:
#                 w.write('{},{}\n'.format(id,p))
        print('inference done!')  
        
        
    elif model_name in sequence_model or model_name in context_aware_model:
        # config, model, dataset 불러오기
        checkpoint = torch.load(model_path)
        config = checkpoint['config']
        config['dataset'] = 'train_data'
        init_seed(config['seed'], config['reproducibility'])
        print("create dataset start!")
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        print("create dataset done!")
        model = get_model(config['model'])(config, test_data.dataset).to(config['device'])
        model.load_state_dict(checkpoint['state_dict'])
        model.load_other_parameter(checkpoint.get('other_parameter'))

        # device 설정
        device = config.final_config_dict['device']

        # user, item id -> token 변환 array
        user_id = config['USER_ID_FIELD']
        item_id = config['ITEM_ID_FIELD']
        user_id2token = dataset.field2id_token[user_id]
        item_id2token = dataset.field2id_token[item_id]

        # user id list
        batch_size = 64
        user_count = len(user_id2token)
        remainder = user_count % batch_size
        padding_size = batch_size - remainder if remainder != 0 else 0
        all_user_list = torch.arange(1, user_count + padding_size+1).reshape(-1, batch_size)
        user_id2token = user_id2token[1:]
        # user, item 길이
        user_len = len(user_id2token)
        item_len = len(item_id2token)

        # user-item sparse matrix
        # matrix = dataset.inter_matrix(form='csr')

        # user id, predict item id 저장 변수
        pred_list = None
        user_list = []

        # model 평가모드 전환
        model.eval()
        matrix = dataset.inter_matrix(form='csr')

        tbar = tqdm(all_user_list, desc=set_color(f"Inference", 'pink'))
        with torch.no_grad():
            for data in tbar:
                if max(data) > len(user_id2token):
                    data = [item for item in data if item <= 1154808]
                    data = torch.tensor(data)
                # interaction 생성
                interaction = dict()
                interaction = Interaction(interaction)
                interaction[user_id] = data
                interaction = interaction.to(device)
                interaction = interaction.repeat_interleave(dataset.item_num)
                interaction.update(
                    test_data.dataset.get_item_feature().to(device).repeat(len(data))
                )

                # user item별 score 예측
                score = model.predict(interaction)
                score = score.view(-1, item_len)

                rating_pred = score.cpu().data.numpy().copy()

                user_index = data.numpy()

                idx = matrix[user_index].toarray() > 0

                rating_pred[idx] = -np.inf
                rating_pred[:, 0] = -np.inf
                ind = np.argpartition(rating_pred, -K)[:, -K:]

                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

                batch_pred_list = ind[
                    np.arange(len(rating_pred))[:, None], arr_ind_argsort
                ]

                if pred_list is None:
                    pred_list = batch_pred_list
                    user_list = user_index
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    user_list = np.append(
                        user_list, user_index, axis=0
                    )

#             result = []
#             for user, pred in zip(user_list, pred_list):
#                 for item in pred:
#                     result.append((int(user_id2token[user]), int(item_id2token[item])))



            # user_list를 key로, pred_list를 value로 갖는 dictionary 생성
            data_dict = {str(user_id): pred_list[i].tolist() for i, user_id in enumerate(user_list)}

            # dictionary를 JSON 형태로 변환
            json_data = json.dumps(data_dict)

            # JSON 문자열을 다시 딕셔너리로 디코딩
            # decoded_data = json.loads(json_data)

            # JSON 데이터를 파일에 저장
            file_path = "inference.json"  # 원하는 파일 경로와 이름 설정
            with open(file_path, 'w') as f:
                f.write(json_data)







if __name__ == "__main__":
    args = parse_args()
    main(args)