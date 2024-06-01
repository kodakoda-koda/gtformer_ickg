import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionNet(nn.Module):
    def __init__(self, d_model):
        super(AttentionNet, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear3 = nn.Linear(d_model, 1)
        self.tanh = nn.Tanh()

    def forward(self, X, Y):
        X_ = self.linear1(X.view(-1, X.shape[-2], X.shape[-1])).permute(0, 2, 1).unsqueeze(-1)
        Y_ = self.linear2(Y.view(-1, X.shape[-2], X.shape[-1])).permute(0, 2, 1).unsqueeze(-2)
        net = self.tanh(torch.matmul(X_, Y_).permute(0, 2, 3, 1))
        out = self.linear3(net).squeeze()

        return out.view(X.shape[0], -1, X.shape[-2], X.shape[-2])


class SpatialAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(SpatialAttentionLayer, self).__init__()
        self.attention_net_A = AttentionNet(d_model)
        self.attention_net_B = AttentionNet(d_model)
        self.attention_net_C = AttentionNet(d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear3 = nn.Linear(d_model, d_model)
        self.linear4 = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-2)
        self.relu = nn.ReLU()
        self.d_model = d_model

    def forward(self, X):
        B_, L, O, D = X.shape

        e = 1e-10
        A = (X) / (X.sum(dim=-1, keepdim=True) + e)  # B, L, O, D
        B = (X) / (X.sum(dim=-2, keepdim=True) + e)
        C = torch.reciprocal(X + e) / (torch.sum(torch.reciprocal(X + e), dim=(-1), keepdim=True))

        V = torch.ones((B_, L, O, self.d_model)).to(X.device)  # B, L, O, d_model

        A_V = torch.matmul(A, V)  # B, L, O, d_model
        B_V = torch.matmul(B, V)
        C_V = torch.matmul(C, V)

        A_net = self.attention_net_A(V, A_V)  # B, L, O, D
        B_net = self.attention_net_B(V, B_V)
        C_net = self.attention_net_C(V, C_V)

        phi = self.softmax(A_net)  # B, L, O, D
        psi = self.softmax(B_net)
        theta = self.softmax(C_net)

        M = (
            self.relu(self.linear1(V))
            + torch.matmul(phi, self.relu(self.linear2(V)))
            + torch.matmul(psi, self.relu(self.linear3(V)))
            + torch.matmul(theta, self.relu(self.linear4(V)))
        )  # B, L, O, d_model

        return M
