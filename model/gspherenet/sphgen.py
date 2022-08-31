# -*- coding: utf-8 -*-
# @Filename: sphgen.py
# @Date: 2022-08-31 09:22
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spherenet import SphereNet


class SphGen(nn.Module):
    def __init__(self, cutoff, num_node_types, num_layers, hidden_channels,
                 int_emb_size, basis_emb_size, out_emb_channels, num_spherical,
                 num_radial, num_flow_layers, device, deg_coeff=0.9, att_heads=4):
        super(SphGen, self).__init__()
        self.device = device
        self.num_node_types = num_node_types

        self.feat_net = SphereNet(cutoff, num_node_types, num_layers, hidden_channels, int_emb_size,
                                  basis_emb_size, out_emb_channels, num_spherical, num_radial)
        node_feat_dim, dist_feat_dim = hidden_channels * 2, hidden_channels * 2
        angle_feat_dim, torsion_feat_dim = hidden_channels * 3, hidden_channels * 4

