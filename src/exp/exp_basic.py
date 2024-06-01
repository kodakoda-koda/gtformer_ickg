import torch


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
