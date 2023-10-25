# GTFormer: a Geospatial Temporal Transformer for Crowd Flow Prediction

This repository is the official implementation of GTFormer: a Geospatial Temporal Transformer for Crowd Flow Prediction. 

<div align="center">
<img src="https://github.com/kodakoda-koda/GTFormer/blob/main/figure/GTFormer.png" width="1000" alt="Figure" title="Architecture of GTFormer">
</div>


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Experiment

1. Download data from [here](https://drive.google.com/drive/folders/1B9WRpkfHn48VfkaHjnErgQ5yb8Vv6PSj?usp=drive_link) and put into /data/NYC or /data/DC.


2. We provide the experiment scripts of all models under the folder ./scripts. You can reproduce the experiment results by:
   ```
   ./scripts/Main/NYC/GTFormer.sh
   ./scripts/Main/DC/GTFormer.sh
   ./scripts/Ablation/Transformer.sh
   ./scripts/Ablation/Attention.sh
   ``` 


## Acknowledgement

We appreciate the following github repo a lot for their valuable code base

https://github.com/thuml/Autoformer

https://github.com/jonpappalord/crowd_flow_prediction

The dataset is provided by:

https://capitalbikeshare.com/system-data

https://citibikenyc.com/system-data

