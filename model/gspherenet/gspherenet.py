# -*- coding: utf-8 -*-
# @Filename: gspherenet.py
# @Date: 2022-08-30 14:24
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import os
import torch
import numpy as np
from .sphgen import SphGen

class G_SphereNet():
    def __init__(self):
        super(G_SphereNet, self).__init__()
        self.model = None

    def get_model(self, model_conf_dict, ckpt_path=None):
        if model_conf_dict['use_gpu'] and not torch.cuda.is_available():
            model_conf_dict['use_gpu'] = False
        self.model = SphGen(**model_conf_dict)
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))

    def load_pretrain_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, loader, lr, wd, max_epochs, config, ckpt_path, save_interval, save_dir):
        self.get_model(config, ckpt_path)
        self.model.train()

