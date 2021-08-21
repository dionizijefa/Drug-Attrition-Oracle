from ogb.utils.features import get_atom_feature_dims
from torch_geometric.nn import EGConv, global_mean_pool
from torch.nn import Sequential, BatchNorm1d, ReLU, Linear, Module, Embedding, ModuleList
from torch.nn.init import xavier_uniform_

full_atom_feature_dims = get_atom_feature_dims()

class AtomEncoder(Module):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = Embedding(dim, emb_dim)
            xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class EGConvNet(Module):
    """Multi aggregators = ['sum', 'mean', 'max'] or
    ['symnorm']"""
    def __init__(self, hidden_channels, num_layers, num_heads, num_bases, aggregator):
        super().__init__()

        self.encoder = AtomEncoder(hidden_channels)

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
            Linear(hidden_channels // 4, 1),
        )

    def forward(self, x, edge_index, batch):
        # EGConv works without any edge features

        x = self.encoder(x)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = h.relu_()
            x = x + h

        x = global_mean_pool(x, batch)

        return self.mlp(x)



