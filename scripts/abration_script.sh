#!/bin/bash

# base
python main.py --data_type Taxi --save_attention True

# use only
python main.py --data_type Taxi --save_outputs True --use_only temporal
python main.py --data_type Taxi --save_outputs True --use_only spatial

# spatial mode
# python main.py --data_type Taxi --save_outputs True --spatial_mode None

# temporal mode
python main.py --data_type Taxi --save_outputs True --temporal_mode None --save_attention True
