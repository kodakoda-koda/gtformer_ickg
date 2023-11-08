import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.lstm1 = nn.LSTM(args.num_tiles**2, args.d_model, batch_first=True)
        self.lstm2 = nn.LSTM(args.d_model, args.d_model, batch_first=True)
        self.linear = nn.Linear(args.d_model, args.num_tiles**2)

    def forward(self, X, _):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination
        B, L, O, D = X.shape
        X = X.view(B, L, O * D)

        lstm1_out, (h, c) = self.lstm1(X)
        lstm2_out, (h, c) = self.lstm2(lstm1_out[:, -1:, :])
        out = self.linear(lstm2_out)
        out = out.view(B, 1, O, D)

        return out
