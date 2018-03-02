import xarray as xr
import torch
from lib.models.torch import column_run

i = snakemake.input

inputs = xr.open_dataset(i.inputs)
forcings = xr.open_dataset(i.forcings)
model = torch.load(i.model)

progs, prec = column_run(model, inputs, forcings)
xr.merge((progs, prec.to_dataset(name="prec")))\
  .to_netcdf(snakemake.output[0])
