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
        temp_out = self.tempLayer(spat_out)
        temp_out = self.bn(temp_out)  # B, L, O, d_model

        out = torch.zeros((B, 1, O, D))
        for o in range(O):
            for d in range(D):
                o_v = temp_out[:, :, o]  # B, L, d_model
                d_v = temp_out[:, :, d].view(B, L, self.d_model, 1)
                o_v = self.linear(o_v).view(B, L, self.d_model, 1).view(B, L, 1, -1)
                out[:, :, o, d] = torch.matmul(o_v, d_v).squeeze()

        return out
