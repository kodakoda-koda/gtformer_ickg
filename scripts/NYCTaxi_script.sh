#!/bin/bash

# MODEL=("AR" "LSTM" "GEML" "CrowdNet" "GTFormer")

# for model in ${MODEL[@]}
# do
# python src/main.py --model $model --data_type Taxi --save_outputs
# done

python src/main.py --model GTFormer --data_type Taxi --save_outputs
