import xarray as xr
import torch
from lib.models.torch import column_run
import logging

i = snakemake.input
RCE = snakemake.params.get('RCE', False)

inputs = xr.open_dataset(i.inputs)
forcings = xr.open_dataset(i.forcings)
model = torch.load(i.model)

if RCE:
    print("Running in RCE mode (time homogeneous forcings)")
    forcings = forcings * 0 + forcings.mean('time')

progs, prec = column_run(model, inputs, forcings)
xr.merge((progs, prec.to_dataset(name="prec")))\
  .to_netcdf(snakemake.output[0])
