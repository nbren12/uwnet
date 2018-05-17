#!/usr/bin/env python
import xarray as xr
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('output')
    parser.add_argument('grd')
    parser.add_argument('-s', '--stat')
    parser.add_argument('-i', '--input')
    parser.add_argument('-t', '--time', type=int)

    return parser.parse_args()

def save_grd(z, path):
    # extrapolate z
    z = np.hstack((z, 2*z[-1] - z[-2]))
    np.savetxt(path, z, fmt="%10.5f")



args = parse_arguments()

# save netcdf file
ds_3d = xr.open_dataset(args.input).isel(time=args.time)
time = float(ds_3d.time)
stat = xr.open_dataset(args.stat).sel(time=time, method='nearest')
ds_3d = ds_3d.assign(Ps=stat.Ps, RHO=stat.RHO)
ds_3d.to_netcdf(args.output)

save_grd(ds_3d.z.data, args.grd)

