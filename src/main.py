import argparse
import os
import random

import numpy as np
import torch

from exp.exp_main import Exp_Main
from utils.exp_utils import set_args


def main():
    fix_seed = 2024
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)

    parser = argparse.ArgumentParser()

    # exp config
    parser.add_argument("--path", type=str, default=".", help="current directory")
    parser.add_argument("--save_path", type=str, default="./results_data")
    parser.add_argument("--model", type=str, default="GTFormer", help="model name")
    parser.add_argument("--sample_time", type=str, default="60min", help="sample time")
    parser.add_argument("--tile_size", type=str, default=None, help="tile size")

    # training config
    parser.add_argument("--itrs", type=int, default=3, help="number of run")
    parser.add_argument("--train_epochs", type=int, default=150, help="epochs")
    parser.add_argument("--patience", type=int, default=10, help="patience of early stopping")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--seq_len", type=int, default=11, help="input sequence length")
    parser.add_argument("--lr", type=float, default=1e-04, help="learning rate")
    parser.add_argument("--city", type=str, default="NYC", help="city name")
    parser.add_argument("--data_type", type=str, default="Bike", help="data type")
    parser.add_argument("--num_tiles", type=int, default=None, help="number of tiles")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout late")
    parser.add_argument("--save_outputs", type=bool, default=False, help="save outputs")
    parser.add_argument("--dtype", type=str, default="fp32", help="dtype")

    # GTFormer config
    parser.add_argument("--d_model", type=int, default=64, help="dimension of model")
    parser.add_argument("--n_head", type=int, default=8, help="number of attention head")
    parser.add_argument("--num_blocks", type=int, default=2, help="number of blocks")
    parser.add_argument("--temporal_mode", type=str, default="BRPE", help='["BRPE", "None"]')
    parser.add_argument("--spatial_mode", type=str, default="AFT-simple", help='["AFT-full", "AFT-simple", "None"]')
    parser.add_argument("--use_only", type=str, default="None", help='["temporal", "spatial", "None"]')
    parser.add_argument("--save_attention", type=bool, default=False, help="save attention")

    args = parser.parse_args()
    args = set_args(args)

    dataset_directory = os.path.join(args.path + "/data/" + args.city + "_" + args.data_type + "/")
    df_path = dataset_directory + "df_grouped_" + args.tile_size + "_" + args.sample_time + ".csv"
    assert os.path.exists(
        df_path
    ), f"df_grouped_{args.tile_size}_{args.sample_time}.csv does not exist in {dataset_directory}"

    print("Args in experiment:")
    print(
        f"""model: {args.model}, city: {args.city}, data type: {args.data_type}, 
        sample time: {args.sample_time}, tile size: {args.tile_size}, num tiles: {args.num_tiles}"""
    )
    if args.model == "GTFormer":
        print(f"temporal_mode: {args.temporal_mode}, spatial_mode: {args.spatial_mode}, use_only: {args.use_only}")

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
