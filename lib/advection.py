"""Script for large-scale advection analysis
"""
import numpy as np
import xarray as xr
from xnoah.xcalc import centderiv


# open dataset
u = xr.open_dataset(snakemake.input.u).U
v = xr.open_dataset(snakemake.input.v).V
w = xr.open_dataset(snakemake.input.w).W

# open dataset
# find variable which isn't p
f = xr.open_dataset(snakemake.input.f)
varname = [v for v in f.data_vars if v!='p'][0]
print(f"Advecting variable {varname}")
f = f[varname]

# compute total derivative
df = u * centderiv(f, dim='x', boundary='periodic')\
    + v * centderiv(f, dim='y', boundary='nearest')\
    + w * centderiv(f, dim='z', boundary='nearest')

# save output
df.to_netcdf(snakemake.output[0])
