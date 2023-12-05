import argparse
import os
import pdb
def parse_args():
    """
    parameter를 전달해주는 함수입니다.
    sweep을 이용하려면 이 함수에 추가를 하셔야 합니다.
    default 값만 사용하신다면 굳이 추가 안하셔도 됩니다.
    예시로 기본 성능이 좋았던 ~~~ 모델 args를 작성하였습니다.
    일단 대표적인 args 몇가지만 작성했고, 추가로 더 필요한 HP는 추가하셔서 사용하시면 됩니다!
    Returns:
        parser : main에 전달될 args
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--mode", default="cont", type=str)
    parser.add_argument("--model_name", default="DCN", type=str)

    parser.add_argument("--dataset_name", default="train_data", type=str)

    parser.add_argument("--config",default = "general.yaml",type=str)

    parser.add_argument("--top_k",default = 10,type=int)
    
    parser.add_argument("--max_len",default = 50,type=int)

    parser.add_argument("--user_lower_bound",default = 0,type=int)
    parser.add_argument("--user_upper_bound",default = 'inf',type=str)
    parser.add_argument("--item_lower_bound",default = 0,type=int)
    parser.add_argument("--item_upper_bound",default = 'inf',type=str)

    parser.add_argument("--filter_inter", default = False, type=bool)
    
    #inference
    
    parser.add_argument("--rank_K", default = 300, type=int, help="# of predict number")
    parser.add_argument("--top_K" , default = 10, type=int)
    models_path = '/home/dhkim/server_front/winery_AI/winery/Recbole/saved'

    # Get a list of all files in the directory
    #files = os.listdir(models_path)
    #files = [file for file in files if os.path.isfile(os.path.join(models_path, file))]
    
    #sorted_files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(models_path, x)), reverse=True)
    #most_recent_model = sorted_files[0]

    parser.add_argument("--saved_model" , default = '/home/dhkim/server_front/winery_AI/winery/Recbole/saved/DCN-latest.pth', type=str,help ="use model")
    args = parser.parse_args()

    return args