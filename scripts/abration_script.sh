#!/bin/bash

# base
python main.py

# use only
python main.py --use_only temporal --num_blocks 1
python main.py --use_only spatial --num_blocks 1

# spatial mode
python main.py --spatial_mode AFT-full
python main.py --spatial_mode None

# temporal mode
python main.py --temporal_mode None