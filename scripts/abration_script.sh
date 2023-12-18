#!/bin/bash

# base
python main.py --data_type Taxi

# use only
python main.py --data_type Taxi --use_only temporal
python main.py --data_type Taxi --use_only spatial

# spatial mode
python main.py --data_type Taxi --spatial_mode AFT-full --dtype bf16
python main.py --data_type Taxi --spatial_mode None --dtype bf16

# temporal mode
python main.py --data_type Taxi -- temporal_mode None
