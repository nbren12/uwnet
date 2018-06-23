import argparse

# plot
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import xarray as xr

from uwnet import model
from uwnet.data import get_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model')
    return parser.parse_args()


# load configuration and process arguments
config = yaml.load(open("config.yaml"))
args = parse_arguments()

# load model
d = torch.load(args.model)
mod = model.MLP.from_dict(d['dict'])


# open training data
def post(x):
    # return x
    # return x.isel(x=slice(0, 1), y=slice(50, 51)).sortby('time')
    return x.isel(x=slice(0, 1), y=slice(24, 40))


paths = config['paths']
data = get_dataset(paths, post=post)

# prepare input for mod
with torch.no_grad():
    batch = {key: torch.from_numpy(val) for key, val in data[:].items()}

# finally run model
    out = mod(batch, n=1)

# save batch to netcdf

def unstack(val):
    val = val.detach().numpy()
    dims = ['xbatch', 'xtime', 'xfeat'][:val.ndim]
    coords = {key: data._ds.coords[key] for key in dims}

    ds = xr.DataArray(val, dims=dims, coords=coords)
    for dim in dims:
        ds = ds.unstack(dim)

    return ds


out_da = {key: unstack(val) for key, val in out.items()}
ds = xr.Dataset(out_da).merge(data.data.rename({'qt': 'QTOBS', 'sl': 'SLOBS'}))
ds.to_netcdf("out.nc")
