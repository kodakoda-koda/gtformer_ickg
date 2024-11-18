import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_provider.create_od_matix import create_od_matrix
from data_provider.data_loader import data_provider
from exp.exp_basic import Exp_Basic
from utils.dataset_utils import get_matrix_mapping, restore_od_matrix, to_io_map
from utils.exp_utils import EarlyStopping, compute_score, output_results, save_output


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.criterion = nn.MSELoss()

    def train(self):
        dataset_directory = os.path.join(self.args.path + "/data/" + self.args.city + "_" + self.args.data_type + "/")
        od_matrix, _, _ = create_od_matrix(dataset_directory, self.args)
        train_loader = data_provider("train", self.args, od_matrix)
        vali_loader = data_provider("val", self.args, od_matrix)

        path = os.path.join(self.args.path + f"/checkpoints/")
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=model_optim, gamma=0.96)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            self.model.train()

            train_loss = []
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs, _, _ = self.model(batch_x)

                loss = self.criterion(outputs, batch_y)
                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())

            my_lr_scheduler.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader)
            print(
                "Epoch: {}, cost time: {:.5f}, Steps: {} | Train Loss: {:.5f} Vali Loss: {:.5f}".format(
                    epoch + 1, time.time() - epoch_time, len(train_loader), train_loss, vali_loss
                )
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopped")
                break

        best_model_path = path + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return

    def vali(self, vali_loader):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs, _, _ = self.model(batch_x)

                loss = self.criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        return total_loss

    def test(self, itr):
        dataset_directory = os.path.join(self.args.path + "/data/" + self.args.city + "_" + self.args.data_type + "/")
        od_matrix, min_tile_id, empty_indices = create_od_matrix(dataset_directory, self.args)
        test_loader = data_provider("test", self.args, od_matrix)

        self.model.load_state_dict(
            torch.load(os.path.join(self.args.path + f"/checkpoints/" + "checkpoint.pth"))
        )
        self.model.eval()

        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs, A_temporal, A_spatial = self.model(batch_x)

                preds.append(outputs.cpu().float().detach().numpy())
                trues.append(batch_y.cpu().float().detach().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        trues_map, preds_map = to_io_map(self.args, trues, preds, min_tile_id, empty_indices)
        od_rmse, od_mae, io_rmse, io_mae = compute_score(trues, preds, trues_map, preds_map)

        # Write results
        output_results(self.args, itr, od_rmse, od_mae, io_rmse, io_mae)

        # save predictions and true values
        save_output(self.args, itr, trues, preds, trues_map, preds_map, A_temporal, A_spatial)
