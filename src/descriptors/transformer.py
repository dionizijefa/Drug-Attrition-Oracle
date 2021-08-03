import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from molecule_attention_transformer import MAT

### Model definition
class Network(torch.nn.Module):
    def __init__(
            self,
            ppi_depth=3):
        super(Network, self).__init__()
        self.MAT = MAT(
            dim_in=11,
            model_dim=1024,
            dim_out=1,
            depth=8,
            Lg=0.33,  # lambda (g)raph - weight for adjacency matrix
            Ld=0.33,  # lambda (d)istance - weight for distance matrix
            La=1,  # lambda (a)ttention - weight for usual self-attention
            dist_kernel_fn='softmax'  # distance kernel fn - either 'exp' or 'softmax'
        )
        self.linear_2 = torch.nn.Linear(321, 1)

    def forward(self, x, mask, adj_mat, dist_mat, descriptors):
        x = self.MAT(x=x,
                     adjacency_mat=adj_mat,
                     distance_mat=dist_mat,
                     mask=mask)
        out = torch.cat((x, descriptors), dim=1)
        out = self.linear_2(out)
        return out

