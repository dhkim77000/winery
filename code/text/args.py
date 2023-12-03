import argparse


def get_args():
    import torch

# Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='total', type=str)
    parser.add_argument("--model_path", 
                    default='/home/dhkim/server_front/winery_AI/winery/code/text/models/model_outputmodel_state_dict_4.pt', 
                    type=str)
    parser.add_argument("--data", default='/home/dhkim/server_front/winery_AI/winery/data/review_df_cleaned.csv', type=str)
    parser.add_argument("--max_len", default = 152, type=int)
    parser.add_argument("--batch_size", default = 16, type=int)
    parser.add_argument("--pool", default = 'mean', type=str)
    #######Data#############################################################################


    args = parser.parse_args()
    return args