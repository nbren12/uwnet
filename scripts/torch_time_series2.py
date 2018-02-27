
"""
{sys.executable} scripts/torch_time_series.py multi \
            --num_epochs 4 --window_size 10 --num_steps 500 --batch_size 100 --learning-rate .010\
--weight_decay 0.00 \
{input} {output}
"""
import xarray as xr
import numpy as np
from sklearn.externals import joblib
import torch
from lib.models.torch import train_multistep_objective
from lib.data import prepare_data
import json

i = snakemake.input
inputs = xr.open_dataset(i.inputs)
forcings = xr.open_dataset(i.forcings)
data = prepare_data(inputs, forcings)
stepper, epoch_data = train_multistep_objective(data, **snakemake.params[0])

torch.save(stepper, snakemake.output.model)
json.dump(epoch_data, open(snakemake.output.json, "w"))
