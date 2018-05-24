import xarray as xr
import torch
from lib.torch import column_run, ForcedStepper



inputs = snakemake.input.inputs
forcings = snakemake.input.forcings
state = snakemake.input.state
nsteps = snakemake.params.get('nsteps', 1)

inputs = xr.open_dataset(inputs)
forcings = xr.open_dataset(forcings)

model = ForcedStepper.from_file(state)
model.eval()

model.nsteps = nsteps
print("nsteps", nsteps)

column_run(model, inputs, forcings)\
    .to_netcdf(snakemake.output[0])
