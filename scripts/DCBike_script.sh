#!/bin/bash

# MODEL=("AR" "LSTM" "GEML" "CrowdNet" "GTFormer")

# for model in ${MODEL[@]}
# do
# python src/main.py --model $model --city DC --data_type Bike --save_outputs
# done

python src/main.py --model GTFormer --city DC --data_type Bike --save_outputs