
"""
{sys.executable} scripts/torch_time_series.py multi \
            --num_epochs 4 --window_size 10 --num_steps 500 --batch_size 100 --learning-rate .010\
--weight_decay 0.00 \
{input} {output}
"""
import numpy as np
from sklearn.externals import joblib
import torch
from lib.models.torch import train_multistep_objective

import dask.bag as db

def train(output):
    print(f"Fitting model {output}")
    data = joblib.load(snakemake.input[0])
    stepper = train_multistep_objective(data, **snakemake.params)
    torch.save(stepper, output)


db.from_sequence(snakemake.output)\
  .map(train)\
  .compute()

