import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.geml_layers import Grid_Embedding


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.spatLayer = Grid_Embedding(args.num_tiles, args.d_model)
        self.tempLayer = nn.LSTM(input_size=args.d_model * 2, hidden_size=args.d_model)
        self.bn = nn.BatchNorm1d(num_features=args.d_model)
        self.linear = nn.Linear(in_features=args.d_model, out_features=args.d_model, bias=True)

        self.d_model = args.d_model

    def forward(self, X, dis_matrix):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination
        B, L, O, D = X.shape
        spat_out = self.spatLayer(X, dis_matrix)
        spat_out = spat_out.view(B * O, L, -1)
        _, (h, _) = self.tempLayer(spat_out)
        temp_out = self.bn(h.permute(0, 2, 1)).view(B, 1, O, -1)  # B, 1, O, d_model

        temp_out = self.linear(temp_out)
        out = torch.matmul(temp_out, temp_out.permute(0, 1, 3, 2))

        return out
