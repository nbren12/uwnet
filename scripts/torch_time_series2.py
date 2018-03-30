"""
{sys.executable} scripts/torch_time_series.py multi \
            --num_epochs 4 --window_size 10 --num_steps 500 --batch_size 100 --learning-rate .010\
--weight_decay 0.00 \
{input} {output}
"""
import xarray as xr
import numpy as np
import torch
from lib.models.torch import train_multistep_objective
from lib.data import prepare_data
import json, sys
from contextlib import redirect_stdout

import logging


def _train(x):
    return x.isel(x=slice(64, None))


def _test(x):
    return x.isel(x=slice(0, 64))


logging.basicConfig(level=logging.DEBUG, filename=snakemake.log[0])

logging.info("Starting training script")

i = snakemake.input
inputs = xr.open_dataset(i.inputs)
forcings = xr.open_dataset(i.forcings)

params = snakemake.params[0]
output_dir = params.pop('output_dir')

train_data = prepare_data(_train(inputs), _train(forcings))
test_data = prepare_data(_test(inputs), _test(forcings))
train_multistep_objective(train_data, test_data, output_dir, **params)

# torch.save(stepper, snakemake.output.model)
# json.dump(epoch_data, open(snakemake.output.json, "w"))
