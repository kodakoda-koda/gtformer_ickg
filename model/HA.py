import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

    def forward(self, X, _):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination
        out = X.mean(dim=1, keepdim=True)

        return out
