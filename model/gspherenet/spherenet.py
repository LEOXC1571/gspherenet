# -*- coding: utf-8 -*-
# @Filename: spherenet.py
# @Date: 2022-08-31 09:47
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import os
import torch
from torch import nn
from torch.nn import Linear, Embedding
from torch_geometric.nn.acts import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
import torch.nn.functional as F
from math import sqrt
#
# from .geometric_computing import xyztodat
# from .features import dist_emb, angle_emb, torsion_emb


class SphereNet(nn.Module):
    def __init__(self, cutoff, num_node_types, num_layers, hidden_channels, int_emb_size, basis_emb_size,
                 out_emb_channels, num_spherical, num_radial, envelope_exponent=5, num_before_skip=1,
                 num_after_skip=2, num_output_layers=3, act=swish):
        super(SphereNet, self).__init__()
        self.cutoff = cutoff
