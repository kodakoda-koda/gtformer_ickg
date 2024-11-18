import os

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
            print("Validation loss decreased ({:.5f} --> {:.5f}).  Saving model ...".format(self.val_loss_min, vali_loss))
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = vali_loss


def compute_score(trues, preds, trues_map, preds_map):
    od_rmse = np.sqrt(mean_squared_error(trues.flatten().reshape(-1, 1), preds.flatten().reshape(-1, 1)))
    od_mae = mean_absolute_error(trues.flatten().reshape(-1, 1), preds.flatten().reshape(-1, 1))
    io_rmse = np.sqrt(
        mean_squared_error(trues_map.flatten().reshape(-1, 1), preds_map.flatten().reshape(-1, 1))
    )
    io_mae = mean_absolute_error(trues_map.flatten().reshape(-1, 1), preds_map.flatten().reshape(-1, 1))

    print("OD flow Prediction")
    print("RMSE Error test: ", od_rmse)
    print("MAE Error test: ", od_mae)
    print("In-Out Flow Prediction")
    print("RMSE Error test: ", io_rmse)
    print("MAE Error test: ", io_mae)

    return od_rmse, od_mae, io_rmse, io_mae


def output_results(args, itr, od_rmse, od_mae, io_rmse, io_mae):
    os.makedirs(args.save_path, exist_ok=True)
    result = {
        "city": args.city,
        "data_type": args.data_type,
        "tile_size": args.tile_size,
        "itr": itr,
        "temporal_mode": args.temporal_mode,
        "spatial_mode": args.spatial_mode,
        "use_only": args.use_only,
        "OD_RMSE": od_rmse,
        "OD_MAE": od_mae,
        "IO_RMSE": io_rmse,
        "IO_MAE": io_mae,
    }
    results_path = os.path.join(args.save_path, "results.csv")
    try:
        results = pd.read_csv(results_path)
    except FileNotFoundError:
        results = pd.DataFrame()
    results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
    results.to_csv(results_path, index=False)


def save_output(args, itr, trues, preds, trues_map, preds_map, A_temporal, A_spatial):
    save_path = args.save_path + f"/{args.city}_{args.data_type}/"
    if args.use_only == "temporal":
        mode = f"{args.use_only}_{args.temporal_mode}/"
    elif args.use_only == "spatial":
        mode = f"{args.use_only}_{args.spatial_mode}/"
    else:
        mode = f"{args.temporal_mode}_{args.spatial_mode}/"
    save_path = save_path + mode

    if args.save_attention:
        if not os.path.exists(save_path + f"/{itr}"):
            os.makedirs(save_path + f"/{itr}")
        np.save(save_path + f"/{itr}/" + "A_temporal.npy", A_temporal.cpu().float().detach().numpy())
        if not args.spatial_mode == "AFT-simple":
            np.save(save_path + f"/{itr}/" + "A_spatial.npy", A_spatial.cpu().float().detach().numpy())

    if args.save_outputs:
        if not os.path.exists(save_path + f"/{itr}"):
            os.makedirs(save_path + f"/{itr}")
        np.save(save_path + f"/{itr}/" + "trues.npy", trues)
        np.save(save_path + f"/{itr}/" + "preds.npy", preds)
        np.save(save_path + f"/{itr}/" + "trues_map.npy", trues_map)
        np.save(save_path + f"/{itr}/" + "preds_map.npy", preds_map)
