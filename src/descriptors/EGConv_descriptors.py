from torch_geometric.nn import EGConv, global_mean_pool
from torch.nn import Sequential, BatchNorm1d, ReLU, Linear, Module, Embedding, ModuleList
from torch.nn.init import xavier_uniform_
from torch import cat


class EGConvNet(Module):
    """Multi aggregators = ['sum', 'mean', 'max'] or
    ['symnorm']"""
    def __init__(self, descriptors_len, hidden_channels, num_layers, num_heads, num_bases, aggregator):
        super().__init__()

        self.lin1 = Linear(27, hidden_channels)
        self.norm1 = BatchNorm1d(hidden_channels)

        self.convs = ModuleList()
        self.norms = ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EGConv(hidden_channels, hidden_channels, aggregator,
                       num_heads, num_bases))
            self.norms.append(BatchNorm1d(hidden_channels))

        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels // 2, bias=False),
            BatchNorm1d(hidden_channels // 2),
            ReLU(inplace=True),
            Linear(hidden_channels // 2, hidden_channels // 4, bias=False),
            BatchNorm1d(hidden_channels // 4),
            ReLU(inplace=True),
        )

        self.linear_2 = Linear((descriptors_len + hidden_channels // 4), 128)
        self.batch_norm_1 = BatchNorm1d(128)
        self.out = Linear((128, 1))

    def forward(self, x, edge_index, batch, descriptors):
        #x = torch.tensor(x).to(torch.int64) za GNN explainer
        x = self.lin1(x)
        x = self.norm1(x)
        x = x.relu_()

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = h.relu_()
            x = x + h

        x = global_mean_pool(x, batch)

        x = self.mlp(x)
        x = cat((x, descriptors), dim=1)
        x = self.linear_2(x)
        x = x.relu_()
        x = self.batch_norm_1(x)
        x = self.out(x)

        return x



