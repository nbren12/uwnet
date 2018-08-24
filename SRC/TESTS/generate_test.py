import xarray as xr
import numpy as np


nx, ny, nz = 10, 11, 12
n = nx * ny * nz

arr = np.arange(n).reshape((nz, ny, nx)).astype(float)
ds  = xr.DataArray(arr, dims=['z', 'y', 'x'], name='U')
ds.to_netcdf("test.nc")
