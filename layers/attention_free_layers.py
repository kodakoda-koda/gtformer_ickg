import math

import torch
import torch.nn.functional as F
from torch import einsum, nn


class AFTFull(nn.Module):
    def __init__(self, seq_len, d_model, n_head, save_outputs):
        super().__init__()

        self.n_head = n_head
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.wbias = nn.Parameter(torch.Tensor(seq_len, seq_len))
        self.save_outputs = save_outputs
        nn.init.xavier_uniform_(self.wbias)

    def forward(self, x):
        B, T, _ = x.shape
        H = self.n_head

        Q = self.query_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)
        K = self.key_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)
        V = self.value_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)
        temp_wbias = self.wbias[:T, :T].unsqueeze(0).unsqueeze(1)

        Q_sig = torch.sigmoid(Q)
        temp = torch.exp(temp_wbias) @ torch.mul(torch.exp(K), V)
        weighted = temp / (torch.exp(temp_wbias) @ torch.exp(K))
        out = torch.mul(Q_sig, weighted)

        out = out.permute(0, 2, 1, 3).view(B, T, -1)

        if self.save_outputs:
            return self.out_projection(out), None
        else:
            return self.out_projection(out), None


class AFTSimple(nn.Module):
    def __init__(self, seq_len, d_model, n_head, save_outputs):
        super().__init__()

        self.n_head = n_head
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.save_outputs = save_outputs

    def forward(self, x):
        B, T, _ = x.shape
        H = self.n_head

        Q = self.query_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)
        K = self.key_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)
        V = self.value_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)

        weights = torch.mul(torch.softmax(K, 1), V).sum(dim=1, keepdim=True)
        Q_sig = torch.sigmoid(Q)
        out = torch.mul(Q_sig, weights)

        out = out.permute(0, 2, 1, 3).view(B, T, -1)

        if self.save_outputs:
            return self.out_projection(out), None
        else:
            return self.out_projection(out), None
