#!/bin/bash

# base
python main.py

# use only
python main.py --use_only temporal
python main.py --use_only spatial

# spatial mode
python main.py --spatial_mode AFT-full # --dtype bf16
python main.py --spatial_mode None # --dtype bf16

# temporal mode
python main.py --temporal_mode None