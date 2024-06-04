#!/bin/bash

# MODEL=("AR" "LSTM" "GEML" "CrowdNet" "GTFormer")

# for model in ${MODEL[@]}
# do
# python src/main.py --model $model --save_outputs
# done

python src/main.py --model GTFormer --save_outputs
