import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, vali_loss, model, path):
        score = -vali_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(vali_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(vali_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, vali_loss, model, path):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {vali_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = vali_loss


def set_args(args):
    if args.use_only in ["temporal", "spatial"]:
        args.save_attention = False

    if args.city == "NYC":
        if args.data_type == "Bike":
            args.num_tiles = 55
            args.tile_size = "1000m"
        else:
            args.num_tiles = 99  # 54
            args.tile_size = "5000m"  # 7500m
    else:
        args.num_tiles = 144
        args.tile_size = "1000m"

    if args.dtype == "bf16":
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float

    return args
