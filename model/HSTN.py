import torch
import torch.nn as nn

from layers.hstn_layers import HSM, HTM


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.hsm = HSM(args)
        self.htm = HTM(args)

    def forward(self, X, A):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination
        B, L, O, D = X.shape

        H = self.hsm(X, A)
        out = self.htm(X, H)

        return out
