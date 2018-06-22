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
    parser.add_argument('config')
    parser.add_argument('output')
    return parser.parse_args()


# load configuration and process arguments
args = parse_arguments()

# load model
config = yaml.load(open(args.config))
d = torch.load(args.model)
mod = model.MLP.from_dict(d['dict'])


# open training data
def post(x):
    # return x
    # return x.isel(x=slice(0, 1), y=slice(50, 51)).sortby('time')
    return x.isel(x=slice(0, 1), y=slice(24, 40))


paths = config['paths']
print("Opening data")
data = get_dataset(paths, post=post)

print("Running model")
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

    if val.shape[-1] == 1:
        dims.pop()
        coords.pop('xfeat')
        val = val[..., 0]
    ds = xr.DataArray(val, dims=dims, coords=coords)
    for dim in dims:
        ds = ds.unstack(dim)

    return ds

print("Reshaping and saving outputs")
out_da = {key: unstack(val) for key, val in out.items()}
ds = xr.Dataset(out_da).merge(data.data.rename({key: key + 'OBS' for key in out}))
ds.to_netcdf(args.output)
