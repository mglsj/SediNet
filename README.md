# SediNet model

SediNet v1.3 codebase modernized.

Original [readme](./sedinet.md)

## Run

Conda

```sh
conda env create -f conda_env/sedinet.yml
```

Activate environment

```sh
conda activate sedinet
```

## Train for global dataset 9 percentile

```sh
CONFIG=config/config_9percentiles.json
python sedinet_train.py -c $CONFIG
```

## Test for global dataset 9 percentile

```sh
WEIGHTS_DIR=grain_size_global/{your_model_file_name.h5}
python sedinet_train.py -c $CONFIG -w $WEIGHTS_DIR
```
