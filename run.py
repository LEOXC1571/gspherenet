# -*- coding: utf-8 -*-
# @Filename: run.py
# @Date: 2022-08-30 14:13
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import os
import json
import yaml
import pickle
import random
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import numpy as np


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from model import G_SphereNet
from dataset import QM93DGEN, collate_fn


def main(args, config):
    if torch.cuda.is_available() and config['gpu_id'] != 'cpu':
        device = torch.device('cuda:' + str(config['gpu_id']))
    else:
        device = torch.device('cpu')

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    if args.train:
        dataset = QM93DGEN()
        split_idx = dataset.get_idx_split('rand_gen')
        train_loader = DataLoader(dataset[split_idx['train']], batch_size=config['batch_size'],
                              shuffle=True, collate_fn=collate_fn)
    else:
        pass


    model = G_SphereNet()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./saved')
    parser.add_argument('--num_mols', type=int, default=1000)
    parser.add_argument('--train', action='store_true', default=True)
    # parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    config = yaml.load(open('config/gspherenet.yaml', 'r'), Loader=yaml.FullLoader)

    main(args, config)

