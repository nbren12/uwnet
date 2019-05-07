import xarray as xr
import os
from os.path import isfile, join

directory = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/OUT_3D/'  # noqa
files = [
    join(directory, f) for f in os.listdir(directory) if
    (isfile(join(directory, f)) and f.endswith('.nc') and (f != '.DS_Store'))
]
ds = xr.auto_combine([
    xr.open_dataset(filename) for filename in files
])
ds.to_netcdf(directory + 'out_3d.nc')
