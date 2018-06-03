"""
{sys.executable} scripts/torch_time_series.py multi \
            --num_epochs 4 --window_size 10 --num_steps 500 --batch_size 100 --learning-rate .010\
--weight_decay 0.00 \
{input} {output}
"""
import xarray as xr
import numpy as np
import torch
from lib.torch import train_multistep_objective, TrainingData
import json, sys
from contextlib import redirect_stdout
import logging

handlers = [logging.FileHandler(snakemake.log[0]), logging.StreamHandler()]
logging.basicConfig(level=logging.DEBUG, handlers=handlers)

params = snakemake.params[0]
i = snakemake.input

logging.info("Starting training script")

# define paths
files = [
    (i.tend, ('FQT', 'FSL')),
    (i.cent, ('QV', 'TABS', 'QN', 'QP', 'QRAD')),
    (i.stat, ('p', 'RHO')),
    (i['2d'], ('LHF', 'SHF', 'SOLIN')),
]

# get training region
north = params.pop('north', 40)
south = params.pop('south', 24)
logging.info(f"Training on data between y-indices {south} and {north}")

def safesel(da, **kwargs):
    kwargs['y'] = slice(24, 40)
    sel = {dim: kwargs[dim] for dim in da.dims
           if dim in kwargs}
    return da.isel(**sel)

def _train(x):
    return safesel(x, x=slice(64, None))

def _test(x):
    return safesel(x, x=slice(0, 64))


logging.info("Loading training data")
train = TrainingData.from_var_files(files, post=_train)
logging.info("Size of training dataset: %.2f MB"%(train.nbytes/1e6))

logging.info("Loading testing data")
test = TrainingData.from_var_files(files, post=_test)
logging.info("Size of testing dataset: %.2f MB"%(test.nbytes/1e6))

logging.info("Calling train_multistep_objective")
output_dir = params.pop('output_dir')
train_multistep_objective(train, test, output_dir, **params)
