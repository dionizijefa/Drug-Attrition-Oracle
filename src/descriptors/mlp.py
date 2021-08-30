from torch.nn import Sequential, BatchNorm1d, ReLU, Linear, Module, ModuleList

class MLPNet(Module):
    def __init__(self, input_dim, hidden_channels):
        super().__init__()

        self.mlp = Sequential(
            Linear(input_dim, hidden_channels // 2, bias=True),
            BatchNorm1d(hidden_channels // 2),
            ReLU(inplace=True),
            Linear(hidden_channels // 2, hidden_channels // 4, bias=True),
            BatchNorm1d(hidden_channels // 4),
            ReLU(inplace=True),
            Linear(hidden_channels // 4, 1),
        )

    def forward(self, descriptors):

        return self.mlp(descriptors)



