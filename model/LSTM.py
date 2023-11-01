import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(args.num_tiles**2, args.d_model, batch_first=True)
        self.linear = nn.Linear(args.d_model, args.num_tiles**2)

    def forward(self, X, _):
        # B: batch size
        # L: sequence length
        # O: num origin
        # D: num destination
        B, L, O, D = X.shape
        X = X.view(B, L, O * D)

        lstm_out, (h, c) = self.lstm(X)
        out = self.linear(lstm_out)
        out = out.view(B, L, O, D)

        return out[:, -1:, :, :]
