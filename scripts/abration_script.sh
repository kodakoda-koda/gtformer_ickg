#!/bin/bash

ONLY=("temporal" "spatial")
for only in ${ONLY[@]}
do
python main.py --data_type Taxi --use_only $only
done

python main.py --data_type Taxi --spatial_mode "AFT-full" --dtype bf16
python main.py --data_type Taxi --spatial_mode "AFT-simple"

T_MODE=("BPRE" "None")
for t_mode in ${T_MODE[@]}
do
python main.py --data_type Taxi --temporal_mode $t_mode
done
