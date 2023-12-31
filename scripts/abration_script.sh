#!/bin/bash

# base
python main.py --data_type Taxi

# use only
python main.py --data_type Taxi --use_only temporal
python main.py --data_type Taxi --use_only spatial

# spatial mode
python main.py --data_type Taxi --spatial_mode None

# temporal mode
python main.py --data_type Taxi --temporal_mode None
