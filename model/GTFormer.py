import torch
import torch.nn as nn

from layers.gtformer_block import GTFormer_block


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.num_blocks = args.num_blocks

        self.blocks = nn.ModuleList([GTFormer_block(args) for _ in range(args.num_blocks)])

        self.out_linear = nn.Linear(args.num_tiles**2, args.num_tiles**2)

    def forward(self, X, key_indices):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination
        B, L, O, D = X.shape

        # X = torch.cat([X, torch.zeros([B, 1, O, D]).to(self.device)], dim=1).view(B, L + 1, O * D)
        X = X.view(B, L, O * D)

        for block in self.blocks:
            X, A_temporal, A_spatial = block(X, key_indices)

        out = self.out_linear(X)
        # out = out.view(B, L + 1, O, D)
        out = out.view(B, L, O, D)

        if self.args.save_outputs:
            return out[:, -1:, :, :], A_temporal, A_spatial
        else:
            return out[:, -1:, :, :]
