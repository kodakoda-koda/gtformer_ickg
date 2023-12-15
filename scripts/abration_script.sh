#!/bin/bash

ONLY=("temporal" "spatial")
for only in ${ONLY[@]}
do
python main.py --use_only $only
done

S_MODE=("AFT-simple" "AFT-full" "None")
for s_mode in ${S_MODE[@]}
do
python main.py --spatial_mode $s_mode
done

T_MODE=("BPRE" "None")
for t_mode in ${T_MODE[@]}
do
python main.py --temporal_mode $t_mode
done
