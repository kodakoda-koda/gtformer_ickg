import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_provider.create_od_matix import create_od_matrix
from data_provider.data_loader import data_provider
from exp.exp_basic import Exp_Basic
from model import AR, GEML, LSTM, CrowdNet, GTFormer
from utils.dataset_utils import get_matrix_mapping, restore_od_matrix, to_2D_map
from utils.exp_utils import EarlyStopping


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            "GTFormer": GTFormer,
            "CrowdNet": CrowdNet,
            "GEML": GEML,
            "LSTM": LSTM,
            "AR": AR,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        return model

    def train(self):
        dataset_directory = os.path.join(self.args.path + "/data/" + self.args.city + "_" + self.args.data_type + "/")
        od_matrix, _, _, _, param = create_od_matrix(dataset_directory, self.args)
        train_loader = data_provider("train", self.args, od_matrix)
        if self.args.model in ["CrowdNet", "GEML"]:
            param = torch.tensor(param).float().to(self.device)

        path = os.path.join(self.args.path + f"/checkpoints_{self.args.model}/")
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.model == "CrowdNet":
            model_optim = torch.optim.RMSprop(self.model.parameters(), lr=self.args.lr, momentum=0.5)
        else:
            model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        criterion = nn.MSELoss()
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=model_optim, gamma=0.96)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.save_outputs:
                    outputs, _, _ = self.model(batch_x, param)
                else:
                    outputs = self.model(batch_x, param)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(od_matrix, param)

            my_lr_scheduler.step()
            print(
                "Epoch: {}, cost time: {}, Steps: {} | Train Loss: {} Vali Loss: {}".format(
                    epoch + 1, time.time() - epoch_time, train_steps, train_loss, vali_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return

    def vali(self, od_matrix, param):
        vali_loader = data_provider("val", self.args, od_matrix)
        total_loss = []
        criterion = nn.MSELoss()
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.save_outputs:
                    outputs, _, _ = self.model(batch_x, param)
                else:
                    outputs = self.model(batch_x, param)

                loss = criterion(outputs, batch_y)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, itr):
        dataset_directory = os.path.join(self.args.path + "/data/" + self.args.city + "_" + self.args.data_type + "/")
        od_matrix, min_tile_id, empty_indices, scaler, param = create_od_matrix(dataset_directory, self.args)
        test_loader = data_provider("test", self.args, od_matrix)
        if self.args.model in ["CrowdNet", "GEML"]:
            param = torch.tensor(param).float().to(self.device)

        self.model.load_state_dict(
            torch.load(os.path.join(self.args.path + f"/checkpoints_{self.args.model}/" + "checkpoint.pth"))
        )

        preds = []
        trues = []

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.save_outputs:
                    outputs, A_temporal, A_spatial = self.model(batch_x, param)
                else:
                    outputs = self.model(batch_x, param)

                preds.append(outputs.cpu().detach().numpy())
                trues.append(batch_y.cpu().detach().numpy())

                if self.args.model == "GTFormer":
                    if self.args.save_outputs:
                        if self.args.use_kvr:
                            A_spatial_ = torch.zeros(
                                (
                                    self.args.batch_size,
                                    self.args.n_head,
                                    self.args.num_tiles**2,
                                    self.args.num_tiles**2,
                                )
                            ).to(self.device)
                            for j in range(self.args.num_tiles**4):
                                A_spatial_[:, :, j, param[j]] = A_spatial[:, :, j, :]

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
        trues = scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)

        # Error of OD flow
        od_rmse_test = np.sqrt(mean_squared_error(trues.flatten().reshape(-1, 1), preds.flatten().reshape(-1, 1)))
        od_mae_test = mean_absolute_error(trues.flatten().reshape(-1, 1), preds.flatten().reshape(-1, 1))

        print("OD flow Prediction")
        print("RMSE Error test: ", od_rmse_test)
        print("MAE Error test: ", od_mae_test)

        # Restore ODmatrirx
        matrix_mapping, x_max, y_max = get_matrix_mapping(self.args)
        trues = restore_od_matrix(trues, empty_indices)
        preds = restore_od_matrix(preds, empty_indices)

        # Conversion OD matrix to IO flow tensor
        trues_map, preds_map = to_2D_map(trues, preds, matrix_mapping, min_tile_id, x_max, y_max, self.args)

        # Erro of IO flow
        io_rmse_test = np.sqrt(
            mean_squared_error(trues_map.flatten().reshape(-1, 1), preds_map.flatten().reshape(-1, 1))
        )
        io_mae_test = mean_absolute_error(trues_map.flatten().reshape(-1, 1), preds_map.flatten().reshape(-1, 1))

        print("In-Out Flow Prediction")
        print("RMSE Error test: ", io_rmse_test)
        print("MAE Error test: ", io_mae_test)

        # Write results
        self.args.path = "/content/drive/MyDrive/2023_Kodama"
        if self.args.model == "GTFormer":
            save_path = os.path.join(
                self.args.path
                + "/results_data/"
                + f"/{self.args.city}_{self.args.data_type}_{self.args.model}_{self.args.spatial_mode}"
            )
        else:
            save_path = os.path.join(
                self.args.path + "/results_data/" + f"/{self.args.city}_{self.args.data_type}_{self.args.model}"
            )
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        f = open(save_path + "/result.txt", "a")
        f.write("itr:{} \n".format(itr + 1))
        f.write(f"args: {self.args}")
        f.write("OD flow prediction:   rmse:{}, mae:{} \n".format(od_rmse_test, od_mae_test))
        f.write("IO flow prediction:   rmse:{}, mae:{} \n".format(io_rmse_test, io_mae_test))
        f.write("\n")
        f.write("\n")
        f.close()

        # save predictions and true values
        if self.args.save_outputs:
            if not os.path.exists(save_path + f"/{itr}"):
                os.makedirs(save_path + f"/{itr}")
            np.save(save_path + f"/{itr}/" + "od_preds.npy", preds)
            np.save(save_path + f"/{itr}/" + "od_trues.npy", trues)
            np.save(save_path + f"/{itr}/" + "io_preds.npy", preds_map)
            np.save(save_path + f"/{itr}/" + "io_trues.npy", trues_map)
            if self.args.model == "GTFormer":
                np.save(save_path + f"/{itr}/" + "A_temporal.npy", A_temporal.cpu().detach().numpy())
                np.save(save_path + f"/{itr}/" + "A_spatial.npy", A_spatial_.cpu().detach().numpy())

        return
