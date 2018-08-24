#!/usr/bin/env python
import argparse
import os
import xarray as xr


def rename_var(z, coords):
    rename_d = {'xc': 'x', 'yc': 'y', 'ys': 'y', 'xs': 'x'}
    rename_d = {key: val for key, val in rename_d.items() if key in z.dims}

    return z.rename(rename_d).assign_coords(**coords)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Create initial condition for'
                                     'coarse resolution SAM')

    parser.add_argument('basedir', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('-t', '--time', type=int, default=0)

    return parser.parse_args()


args = parse_arguments()

time = args.time
stagger_path = os.path.join(args.basedir, "stagger", "3d", "all.nc")
center_path = os.path.join(args.basedir, "coarse", "3d", "all.nc")
stat_path = os.path.join(args.basedir, "stat.nc")

cent = xr.open_dataset(center_path, engine='netcdf4').isel(time=time)

time = cent.time
stag = (xr.open_dataset(stagger_path).sel(time=time)
        .apply(lambda x: rename_var(x, cent.coords)))
stat = xr.open_dataset(stat_path)

ic = xr.Dataset({
    'U': stag.U,
    'V': stag.V,
    'W': cent.W,
    'QV': cent.QV,
    'TABS': cent.TABS,
    'QN': cent.QN,
    'QP': cent.QP,
    'RHO': stat.RHO.sel(time=time),
    'Ps': stat.Ps.sel(time=time)
})

ic.to_netcdf(args.output)
