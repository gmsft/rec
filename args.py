import argparse
import os

import torch
def get_args():
    """
    manage the parameters here.
    :return:
    """
    # print(os.path.realpath(r"."))
    parse = argparse.ArgumentParser()
    parse.add_argument('--csv_path', default=f'{os.path.realpath(".")}/datasets/ml-100k/u.data', type=str, help='the path of data file')
    parse.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='gpu or cpu')
    parse.add_argument('--epoch', default=30, type=int)
    parse.add_argument('--latent_dim', default=500, type=int)
    parse.add_argument('--weight_decay', default=1e-5, type=float)
    parse.add_argument('--dropout', default=1e-5, type=float)
    parse.add_argument('--batch_size', default=1000, type=int)
    parse.add_argument('--lr', default=2e-3, type=float, help='learning rate')
    return parse.parse_args()