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
        temp_out, (h, c) = self.tempLayer(spat_out)
        temp_out = self.bn(temp_out.permute(0, 2, 1)).view(B, L, O, -1)  # B, L, O, d_model

        out = torch.zeros((B, 1, O, D))
        for o in range(O):
            for d in range(D):
                o_v = temp_out[:, -1:, o]  # B, 1, d_model
                d_v = temp_out[:, -1:, d].view(B, 1, self.d_model, 1)
                o_v = self.linear(o_v).view(B, 1, self.d_model, 1).view(B, 1, 1, -1)
                out[:, :, o, d] = torch.matmul(o_v, d_v).squeeze()  # BLをB1に代入しようとしている

        return out
