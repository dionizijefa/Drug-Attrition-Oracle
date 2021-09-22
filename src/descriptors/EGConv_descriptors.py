from torch_geometric.nn import EGConv, global_mean_pool
from torch.nn import Sequential, BatchNorm1d, ReLU, Linear, Module, Embedding, ModuleList
from torch.nn.init import xavier_uniform_
from torch import cat


class EGConvDescriptors(Module):
    """Multi aggregators = ['sum', 'mean', 'max'] or
    ['symnorm']"""

    def __init__(self, hidden_channels, num_layers, num_heads, num_bases, aggregator, descriptors_len, options):
        super().__init__()
        self.options = options

        self.lin1 = Linear(27, hidden_channels)
        self.norm1 = BatchNorm1d(hidden_channels)

        self.convs = ModuleList()
        self.norms = ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EGConv(hidden_channels, hidden_channels, aggregator,
                       num_heads, num_bases))
            self.norms.append(BatchNorm1d(hidden_channels))

        if self.options == "hidden_descriptors":
            self.descriptors_mlp = Sequential(
                Linear(descriptors_len, hidden_channels, bias=False),
                BatchNorm1d(hidden_channels),
                ReLU(inplace=True),
                Linear(hidden_channels, hidden_channels // 4, bias=False),
                BatchNorm1d(hidden_channels // 4),
                ReLU(inplace=True),
            )

            self.mlp = Sequential(
                Linear(hidden_channels, hidden_channels // 2, bias=False),
                BatchNorm1d(hidden_channels // 2),
                ReLU(inplace=True),
                Linear(hidden_channels // 2, hidden_channels // 4, bias=False),
                BatchNorm1d(hidden_channels // 4),
                ReLU(inplace=True),
            )

            self.lin2 = Linear((hidden_channels // 2), 128) #concat layer
            self.bn2 = BatchNorm1d(128)
            self.act2 = ReLU()
            self.out = Linear(128, 1)

        if self.options == "concat_descriptors":
            self.mlp = Sequential(
                Linear(hidden_channels, hidden_channels // 2, bias=False),
                BatchNorm1d(hidden_channels // 2),
                ReLU(inplace=True),
                Linear(hidden_channels // 2, hidden_channels // 4, bias=False),
                BatchNorm1d(hidden_channels // 4),
                ReLU(inplace=True),
            )

            self.lin2 = Linear(descriptors_len+(hidden_channels // 4), 128)
            self.bn2 = BatchNorm1d(128)
            self.act2 = ReLU()
            self.out = Linear(128, 1)

        if self.options == "average_outputs":
            self.descriptors_mlp = Sequential(
                Linear(descriptors_len, hidden_channels, bias=False),
                BatchNorm1d(hidden_channels),
                ReLU(inplace=True),
                Linear(hidden_channels, hidden_channels // 4, bias=False),
                BatchNorm1d(hidden_channels // 4),
                ReLU(inplace=True),
                Linear(hidden_channels // 4, 1),
                ReLU(inplace=True)
            )

            self.mlp = Sequential(
                Linear(hidden_channels, hidden_channels // 2, bias=False),
                BatchNorm1d(hidden_channels // 2),
                ReLU(inplace=True),
                Linear(hidden_channels // 2, hidden_channels // 4, bias=False),
                BatchNorm1d(hidden_channels // 4),
                ReLU(inplace=True),
                Linear(hidden_channels // 4, 1),
                ReLU(inplace=True),
            )

            self.out = Linear(2, 1)

        if self.options == "concat_early":
            self.mlp = Sequential(
                Linear(hidden_channels+descriptors_len, hidden_channels, bias=False),
                BatchNorm1d(hidden_channels),
                ReLU(inplace=True),
                Linear(hidden_channels, hidden_channels // 2, bias=False),
                BatchNorm1d(hidden_channels // 2),
                ReLU(inplace=True),
                Linear(hidden_channels // 2, hidden_channels // 4, bias=False),
                BatchNorm1d(hidden_channels // 4),
                ReLU(inplace=True),
                Linear(hidden_channels //4 , 1),
            )

    def forward(self, x, edge_index, batch, descriptors):
        # x = torch.tensor(x).to(torch.int64) for GNN explainer
        x = self.lin1(x)
        x = self.norm1(x)
        x = x.relu_()

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = h.relu_()
            x = x + h

        x = global_mean_pool(x, batch)

        if self.options == 'hidden_descriptors':
            x = self.mlp(x)
            descriptors = self.descriptors_mlp(descriptors)
            x = cat((x, descriptors), dim=1)
            x = self.act2(self.bn2(self.lin2(x)))

            return self.out(x)

        if self.options == 'concat_descriptors':
            x = self.mlp(x)
            x = cat((x, descriptors), dim=1)
            x = self.act2(self.bn2(self.lin2(x)))

            return self.out(x)

        if self.options == 'average_outputs':
            x = self.mlp(x)
            descriptors = self.descriptors_mlp(descriptors)

            x = cat((x, descriptors), dim=1)

            return self.out(x)

        if self.options == 'concat_early':
            x = cat((x, descriptors), dim=1)

            return self.mlp(x)



