import xarray as xr
import torch
from lib.torch import column_run, ForcedStepper
import logging

i = snakemake.input
RCE = snakemake.params.get('RCE', False)

inputs = xr.open_dataset(i.inputs)
forcings = xr.open_dataset(i.forcings)


model = ForcedStepper.load_from_saved(torch.load(i.state))
model.eval()


if RCE:
    print("Running in RCE mode (time homogeneous forcings)")
    forcings = forcings * 0 + forcings.mean('time')

progs, prec = column_run(model, inputs, forcings)
xr.merge((progs, prec.to_dataset(name="prec")))\
  .to_netcdf(snakemake.output[0])
