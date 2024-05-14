import torch
import torch.nn as nn

from layers.bgarn_layers import AttentionNet, SpatialAttentionLayer


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.attention_net = AttentionNet(args.d_model)
        self.spatial_layer = SpatialAttentionLayer(args.d_model)
        self.temporal_layer = nn.LSTM(input_size=args.d_model, hidden_size=args.d_model, num_layers=1, batch_first=True)
        self.linear = nn.Linear(args.d_model, args.d_model)
        self.bn = nn.BatchNorm1d(args.d_model)
        self.relu = nn.ReLU()
        self.d_model = args.d_model

    def forward(self, X, _):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination
        B, L, O, D = X.shape

        M = self.spatial_layer(X)  # B, L, O, d_model
        M_T = self.temporal_layer(M.view(-1, L, self.d_model))[0][:, -1, :].view(B, O, self.d_model)  # B, O, d_model
        M_T = self.bn(M_T.permute(0, 2, 1)).permute(0, 2, 1)  # B, O, d_model

        M_T = self.relu(self.linear(M_T))  # B, O, d_model
        G = self.attention_net(M_T, M_T)  # B, 1, O, D
        out = G + X.mean(dim=1, keepdim=True)  # B, 1, O, D

        return out
