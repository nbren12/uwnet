import numpy as np


def compute_dp(p):
    p_ghosted = np.hstack((2 * p[0] - p[1], p, 0))
    p_interface = (p_ghosted[1:] + p_ghosted[:-1]) / 2
    dp = -np.diff(p_interface)

    return dp


def snakerun():
    import xarray as xr
    p = xr.open_dataset(snakemake.input[0]).p
    dp = compute_dp(p)
    w = xr.DataArray(dp,  p.coords)
    w.attrs['units'] = p.units
    w.to_dataset(name='w')\
     .to_netcdf(snakemake.output[0])


try:
    snakemake
except NameError:
    pass
else:
    snakerun()
