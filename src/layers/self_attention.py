import math

import torch
import torch.nn as nn


class Temporal_SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, save_attention):
        super(Temporal_SelfAttention, self).__init__()

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)

        self.n_head = n_head
        self.save_attention = save_attention

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_head

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, L, H, -1)
        values = self.value_projection(x).view(B, L, H, -1)

        scale = 1.0 / math.sqrt(queries.shape[-1])

        scores = torch.einsum("blhd,bshd->bhls", queries, keys)

        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        out = V.contiguous().view(B, L, -1)

        if self.save_attention:
            return out, A
        else:
            return out, None


class Relative_Temporal_SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, len_, save_attention):
        super(Relative_Temporal_SelfAttention, self).__init__()

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)

        # Relative Postion Embedding E
        self.e_projection = nn.Linear(d_model // n_head, len_, bias=False)

        self.n_head = n_head
        self.save_attention = save_attention

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_head

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, L, H, -1)
        values = self.value_projection(x).view(B, L, H, -1)

        # QE
        qe = self.e_projection(queries).permute(0, 2, 1, 3)

        # Compute S^rel
        m = nn.ReflectionPad2d((0, L - 1, 0, 0))
        qe = nn.functional.pad(m(qe), (0, 1, 0, L - 1))
        qe = qe.reshape(B, H, qe.shape[-1], qe.shape[-2])
        s_rel = qe[:, :, :L, L - 1 :]

        scale = 1.0 / math.sqrt(queries.shape[-1])

        scores = torch.einsum("blhd,bshd->bhls", queries, keys)

        A = torch.softmax(scale * (scores + s_rel), dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        out = V.contiguous().view(B, L, -1)

        if self.save_attention:
            return out, A
        else:
            return out, None


class Spatial_SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, save_attention):
        super(Spatial_SelfAttention, self).__init__()

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.n_head = n_head
        self.save_attention = save_attention

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_head

        queries = self.query_projection(x).view(B, L, H, -1)
        keys = self.key_projection(x).view(B, L, H, -1)
        values = self.value_projection(x).view(B, L, H, -1)

        scale = 1.0 / math.sqrt(queries.shape[-1])

        scores = torch.einsum("blhd,bshd->bhls", queries, keys)

        A = torch.softmax(scale * scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        out = V.contiguous().view(B, L, -1)

        if self.save_attention:
            return out, A
        else:
            return out, None


class AFTSimple(nn.Module):
    def __init__(self, num_tiles, d_model, n_head, save_attention):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)

        self.save_attention = save_attention

    def forward(self, x):
        B, T, _ = x.shape
        H = self.n_head

        queries = self.query_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)
        keys = self.key_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)
        values = self.value_projection(x).view(B, T, H, -1).permute(0, 2, 1, 3)

        weights = torch.mul(torch.softmax(keys, 1), values).sum(dim=1, keepdim=True)
        queries_sig = torch.sigmoid(queries)
        out = torch.mul(queries_sig, weights)

        out = out.permute(0, 2, 1, 3).view(B, T, -1)

        return out, None
