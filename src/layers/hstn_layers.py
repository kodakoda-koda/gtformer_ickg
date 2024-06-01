import math

import torch
import torch.nn as nn

from layers.embed import TokenEmbedding_temporal


class HSM(nn.Module):
    def __init__(self, args):
        super(HSM, self).__init__()
        self.linear1 = nn.Linear(args.num_tiles, args.d_model)
        self.arus = nn.ModuleList([ARU(args) for _ in range(args.num_blocks)])
        self.linear2 = nn.Linear(args.num_tiles, args.d_model)
        self.frus = nn.ModuleList([FRU(args) for _ in range(args.num_blocks)])
        self.linear3 = nn.Linear(args.num_tiles, args.d_model)
        self.transformer_encoder = TransformerEncoder(args)
        self.fusion = nn.Linear(3 * args.d_model, args.d_model)

        self.gru = nn.GRU(
            input_size=args.d_model * args.num_tiles, hidden_size=args.d_model, num_layers=1, batch_first=True
        )

    def forward(self, X, A):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination
        B, L, O, D = X.shape

        O_a = self.linear1(X)  # B, L, O, d_model
        for aru in self.arus:
            O_a = aru(O_a, A)

        O_f = self.linear2(X)  # B, L, O, d_model
        for fru in self.frus:
            O_f = fru(O_f, A)

        O_i = self.linear3(X).view(B * L, O, -1)  # B * L, O, d_model
        O_i = self.transformer_encoder(O_i).view(B, L, O, -1)  # B, L, O, d_model

        O = self.fusion(torch.cat([O_a, O_f, O_i], dim=-1))  # B, L, O, d_model

        H, _ = self.gru(O.view(O.shape[0], O.shape[1], -1))  # B, L, d_model

        return H


class HTM(nn.Module):
    def __init__(self, args):
        super(HTM, self).__init__()
        self.dlu = DLU(args)
        self.slu = SLU(args)
        self.tg = TG(args)

    def forward(self, X, H):
        F_D = self.dlu(H)
        F_S = self.slu(X)
        Y = self.tg(X, F_S, F_D)

        return Y


class DLU(nn.Module):
    def __init__(self, args):
        super(DLU, self).__init__()
        self.linear = nn.Linear(2 * args.d_model, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H):
        # H: B, L, d_model

        e = torch.concat([H, H[:, -1:].repeat(1, H.shape[1], 1)], dim=-1)  # B, L, 2 * d_model
        E = self.relu(self.linear(e))  # B, L, 1
        A = self.softmax(E)  # B, L, 1
        F_D = torch.matmul(A.transpose(1, 2), H).squeeze()  # B, d_model

        return F_D


class SLU(nn.Module):
    def __init__(self, args):
        super(SLU, self).__init__()
        self.embed = TokenEmbedding_temporal(args.num_tiles**2, args.d_model)
        self.transformer_encoder = TransformerEncoder(args)
        self.linear = nn.Linear(args.d_model, args.num_tiles**2)

    def forward(self, X):
        B, L, O, D = X.shape

        X = X.view(X.shape[0], X.shape[1], -1)  # B, L, O * D
        X = self.embed(X)  # B, L, d_model

        X_k = self.transformer_encoder(X)  # B, L, d_model
        F_S = self.linear(X_k)  # B, L, O * D
        F_S = F_S.sum(dim=1)  # B, O * D

        return F_S.view(B, O, D)


class TG(nn.Module):
    def __init__(self, args):
        super(TG, self).__init__()
        self.linear1 = nn.Linear((args.num_tiles**2) * 2, args.d_model)
        self.linear2 = nn.Linear(args.d_model * 2, args.d_model)
        self.gru = nn.GRU(input_size=args.d_model, hidden_size=args.d_model, num_layers=1, batch_first=True)
        self.linear3 = nn.Linear(args.d_model, args.num_tiles**2)
        self.relu = nn.ReLU()

    def forward(self, X, F_S, F_D):
        B, L, O, D = X.shape

        X = X[:, -1, :, :].view(B, O * D)
        F_S = F_S.view(B, O * D)
        FN = self.linear1(torch.concat([F_S, X], dim=-1))  # B, d_model

        FN = self.linear2(torch.concat([FN, F_D], dim=-1))  # B, d_model

        out, _ = self.gru(FN.unsqueeze(1))  # B, 1, d_model
        out = self.relu(self.linear3(out))  # B, 1, O * D

        return out.view(B, 1, O, D)


class ARU(nn.Module):
    def __init__(self, args):
        super(ARU, self).__init__()
        self.linear = nn.Linear(args.d_model, args.d_model)
        self.relu = nn.ReLU()

    def forward(self, X, A):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination

        Y = self.relu(self.linear(torch.matmul(A, X)))  # B, L, O, d_model

        return Y


class FRU(nn.Module):
    def __init__(self, args):
        super(FRU, self).__init__()
        self.linear = nn.Linear(args.d_model, args.d_model)
        self.relu = nn.ReLU()

    def forward(self, X, A):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination

        Y = self.relu(self.linear(torch.matmul(A, X)))  # B, L, O, d_model

        return Y


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()

        self.query_projection = nn.Linear(args.d_model, args.d_model, bias=False)
        self.key_projection = nn.Linear(args.d_model, args.d_model, bias=False)
        self.value_projection = nn.Linear(args.d_model, args.d_model, bias=False)
        self.linear = nn.Linear(args.d_model, args.d_model)

        self.n_head = args.n_head

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
        out = self.linear(V.contiguous().view(B, L, -1))

        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(args)
        self.conv1 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_model * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.d_model * 4, out_channels=args.d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(args.d_model)
        self.norm2 = nn.LayerNorm(args.d_model)
        self.dropout1 = nn.Dropout(args.dropout)
        self.dropout2 = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        new_x = self.attention(x)
        x = x + self.dropout1(new_x)

        y = x = self.norm1(x)
        y = self.dropout2(self.relu(self.conv1(y.transpose(-1, 1))))
        y = self.conv2(y).transpose(-1, 1)

        return self.norm2(x + y)


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.enc_layer = nn.ModuleList([TransformerEncoderLayer(args) for _ in range(args.num_blocks)])
        self.norm = nn.LayerNorm(args.d_model)

    def forward(self, x):
        for layer in self.enc_layer:
            x = layer(x)
            x = self.norm(x)

        return x
