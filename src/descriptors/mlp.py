import torch
from ogb.utils.features import get_atom_feature_dims
from torch_geometric.nn import EGConv, global_mean_pool
from torch.nn import Sequential, BatchNorm1d, ReLU, Linear, Module, ModuleList

class MLPNet(Module):
    def __init__(self, input_dim, hidden_channels):
        super().__init__()

        self.mlp = Sequential(
            Linear(input_dim, 1024 // 2, bias=False),
            BatchNorm1d(hidden_channels // 2),
            ReLU(inplace=True),
            Linear(hidden_channels // 2, hidden_channels // 4, bias=False),
            BatchNorm1d(hidden_channels // 4),
            ReLU(inplace=True),
            Linear(hidden_channels // 4, 1),
        )

    def forward(self, descriptors):

        return self.mlp(descriptors)



