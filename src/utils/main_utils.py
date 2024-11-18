import torch

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
