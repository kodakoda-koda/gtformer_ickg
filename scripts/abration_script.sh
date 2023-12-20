#!/bin/bash

# base
python main.py

# use only
python main.py --use_only temporal
python main.py --use_only spatial

# spatial mode
# python main.py --spatial_mode AFT-full
# python main.py --spatial_mode None

# temporal mode
python main.py --temporal_mode None

# base
python main.py --data_type Taxi

# use only
python main.py --data_type Taxi --use_only temporal
python main.py --data_type Taxi --use_only spatial

# spatial mode
# python main.py --data_type Taxi --spatial_mode AFT-full
# python main.py --data_type Taxi --spatial_mode None

# temporal mode
python main.py --data_type Taxi --temporal_mode None
