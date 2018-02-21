import xarray as xr
import torch
from lib.plots.single_column_initial_value import column_run

i = snakemake.input

inputs = xr.open_dataset(i.inputs).stack(batch=['x', 'y'])
forcings = xr.open_dataset(i.forcings).stack(batch=['x', 'y'])
model = torch.load(i.model)

progs, prec = column_run(model, inputs, forcings)
xr.merge((progs, prec.to_dataset(name="prec")))\
  .unstack('batch')\
  .to_netcdf(snakemake.output[0])
