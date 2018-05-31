import xarray as xr
from lib.data import inputs_and_forcings_sam

inputs, forcings = inputs_and_forcings_sam(snakemake.input.d3, snakemake.input.d2,
                                           snakemake.input.stat)
o = snakemake.output
inputs.to_netcdf(o.inputs)
forcings.to_netcdf(o.forcings)
