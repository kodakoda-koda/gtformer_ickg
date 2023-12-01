import argparse
import os
import random

import numpy as np
import torch

from data_provider.read_geodataframe import load_dataset
from exp.exp_main import Exp_Main


def main():
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="Crowd Prediction")

    # exp config
    parser.add_argument("--path", type=str, default=".", help="current directory")
    parser.add_argument("--save_path", type=str, default="/content/drive/MyDrive/2023_Kodama/results_data")
    parser.add_argument("--model", type=str, default="GTFormer", help="model name")
    parser.add_argument("--sample_time", type=str, default="60min", help="sample time")
    parser.add_argument("--tile_size", type=str, default=None, help="tile size")

    parser.add_argument("--itrs", type=int, default=1, help="number of run")
    parser.add_argument("--train_epochs", type=int, default=150, help="epochs")  # 30 GTFormer 150 CrowdNet
    parser.add_argument("--patience", type=int, default=10, help="patience of early stopping")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--seq_len", type=int, default=11, help="input sequence length")
    parser.add_argument("--lr", type=int, default=1e-04, help="learning rate")
    parser.add_argument("--city", type=str, default="NYC", help="city name")
    parser.add_argument("--data_type", type=str, default="Bike", help="data type")
    parser.add_argument("--num_tiles", type=int, default=None, help="number of tiles")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout late")
    parser.add_argument("--save_outputs", type=bool, default=False, help="save outputs")

    # GTFormer config
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--temporal_mode", type=str, default="BRPE", help='["BRPE", "None"]')
    parser.add_argument("--spatial_mode", type=str, default="AFT-simple", help='["AFT-full", "AFT-simple", "None"]')
    parser.add_argument("--use_only", type=str, default=None, help='["temporal", "spatial", "None"]')
    parser.add_argument("--save_attention", type=bool, default=False, help="save attention")

    args = parser.parse_args(args=[])
    if args.use_only:
        args.save_attention = False

    if args.city == "NYC":
        if args.data_type == "Bike":
            args.num_tiles = 55
            args.tile_size = "1000m"
        else:
            args.num_tiles = 99
            args.tile_size = "5000m"
    else:
        args.num_tiles = 68
        args.tile_size = "2000m"

    dataset_directory = os.path.join(args.path + "/data/" + args.city + "_" + args.data_type + "/")
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)
    if not os.path.isfile(dataset_directory + "df_grouped_" + args.tile_size + "_" + args.sample_time + ".csv"):
        load_dataset(args.city, args.data_type, args.tile_size, args.sample_time, dataset_directory)

    print("Args in experiment:")
    print(
        f"model: {args.model}, city: {args.city}, data type: {args.data_type}, sample time: {args.sample_time}, tile size: {args.tile_size}, num tiles: {args.num_tiles}"
    )
    if args.model == "GTFormer":
        print(f"num_blocks: {args.num_blocks}, temporal_mode: {args.temporal_mode}, spatial_mode: {args.spatial_mode}")

    Exp = Exp_Main

    for itr in range(args.itrs):
        print("\n")
        print("------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------")
        print(f"itr : {itr+1}")

        exp = Exp(args)  # set experiments
        print(">>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.train()

        print(">>>>>>>testing : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(itr)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
