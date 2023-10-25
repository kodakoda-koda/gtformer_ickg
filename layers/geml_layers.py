import torch
import torch.nn as nn


class Grid_Embedding(nn.Module):
    def __init__(self, num_tiles, d_model):
        super(Grid_Embedding, self).__init__()
        self.linear = nn.Linear(num_tiles * 2, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

        self.d_model = d_model

    def forward(self, X, dis_matrix):
        B, L, O, D = X.shape
        sem_neibor = X > 0
        geo_neibor = dis_matrix <= 2

        X_ = torch.zeros((B, L, O, O + D))
        for i in range(O):
            X_[:, :, i] = torch.cat((X[:, :, i], X[:, :, :, i]), dim=-1)

        Y = self.linear(X_)

        geo_out = torch.zeros((B, L, O, self.d_model))
        sem_out = torch.zeros((B, L, O, self.d_model))
        for i in range(Y.shape[2]):
            index = (geo_neibor[i, :] == True).nonzero().squeeze()
            dis_w = torch.sqrt(dis_matrix[i, index]) / torch.sqrt(dis_matrix[i]).sum()
            f_j = torch.mul(Y[:, :, index].permute(0, 1, 3, 2), dis_w).permute(0, 1, 3, 2)
            a = Y[:, :, i] + f_j.sum(dim=2)
            a = self.linear2(a)
            geo_out[:, :, i] = a

            index_o = sem_neibor[:, :, i]
            index_d = sem_neibor[:, :, :, i]
            index = index_o | index_d
            tile_deg = X_.sum(dim=-1)
            sum_deg = tile_deg.sum(dim=-1)
            tile_deg[~index] = 0
            deg_w = tile_deg / sum_deg.unsqueeze(-1).expand(tile_deg.shape)
            b = Y * deg_w.unsqueeze(-1)
            b = b.sum(dm=2)
            b = Y[:, :, i] + b
            b = self.linear3(b)
            sem_out[:, :, i] = b

        v = torch.cat((geo_out, sem_out), dim=-1)

        return v  # (B, L, O, d_model*2)
