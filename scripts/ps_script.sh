#!/bin/sh

PS = ("parallel" "series_t" "series_s")

for ps in PS
do

python main.py --connection $ps

done
