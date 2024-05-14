import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionNet(nn.Module):
    def __init__(self, d_model):
        super(AttentionNet, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()

    def forward(self, X, Y):
        X_ = self.linear1(X.view(-1, X.shape[-2], X.shape[-1])).permute(0, 2, 1).unsqueeze(-1)
        Y_ = self.linear1(Y.view(-1, X.shape[-2], X.shape[-1])).permute(0, 2, 1).unsqueeze(-2)
        net = torch.matmul(X_, Y_).permute(0, 2, 3, 1)
        out = self.relu(self.linear2(net)).squeeze()

        return out.view(X.shape[0], -1, X.shape[-2], X.shape[-2])


class SpatialAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(SpatialAttentionLayer, self).__init__()
        self.attention_net = AttentionNet(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, X):
        B, L, O, D = X.shape

        A = (X + 1e-5) / (X.sum(dim=-1, keepdim=True) + 1e-5)  # B, L, O, D
        B = (X + 1e-5) / (X.sum(dim=-2, keepdim=True) + 1e-5)
        C = torch.reciprocal(X + 1e-5) / (torch.sum(torch.reciprocal(X + 1e-5), dim=(-1), keepdim=True) + 1e-5)

        V = torch.ones(B, L, O, self.d_model).to(X.device)  # B, L, O, d_model

        A_V = torch.matmul(A, V)  # B, L, O, d_model
        B_V = torch.matmul(B, V)
        C_V = torch.matmul(C, V)

        A_net = self.attention_net(V, A_V)  # B, L, O, D
        B_net = self.attention_net(V, B_V)
        C_net = self.attention_net(V, C_V)

        phi = self.softmax(A_net)  # B, L, O, D
        psi = self.softmax(B_net)
        theta = self.softmax(C_net)

        V_ = self.linear(V)  # B, L, O, d_model

        M = V_ + torch.matmul(phi, V_) + torch.matmul(psi, V_) + torch.matmul(theta, V_)  # B, L, O, d_model

        return M
