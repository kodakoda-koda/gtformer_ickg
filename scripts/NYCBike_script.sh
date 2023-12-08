#!/bin/sh

MODEL = ("AR" "LSTM" "GEML" "CrowdNet" "GTFormer")

for model in MODEL
do

python main.py --model $model

done
