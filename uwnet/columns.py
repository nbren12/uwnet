import argparse

# plot
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import yaml
from torch.utils.data import DataLoader

import xarray as xr

from uwnet import model
from uwnet.data import get_dataset, XRTimeSeries
from uwnet.utils import concat_dicts


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('model')
    parser.add_argument('data')
    parser.add_argument('output')
    return parser.parse_args()


# load configuration and process arguments
args = parse_arguments()

# load model
d = torch.load(args.model)
mod = model.MLP.from_dict(d['dict'])


# open training data
def post(x):
    return x
    # return x.isel(x=slice(0, 1), y=slice(50, 51)).sortby('time')
    # return x.isel(x=slice(0, 1), y=slice(24, 40))


print("Opening data")
ds = xr.open_zarr(args.data).isel(x=slice(0,8))
data = XRTimeSeries(ds.load(), [['time'], ['x', 'y'], ['z']])
loader = DataLoader(data, batch_size=128, shuffle=False)

constants = data.torch_constants()

print("Running model")
# prepare input for mod
outputs = []
with torch.no_grad():
    for batch in tqdm(loader):
        batch.update(constants)
        out = mod(batch, n=1)
        outputs.append(out)



# concatenate outputs
out = concat_dicts(outputs, dim=0)

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

    # transpose dims
    dim_order = [dim for dim in ['time', 'z', 'y', 'x']
                 if dim in ds.dims]
    ds = ds.transpose(*dim_order)

    return ds

print("Reshaping and saving outputs")
out_da = {key: unstack(val) for key, val in out.items()}
ds = xr.Dataset(out_da).merge(data.data.rename({key: key + 'OBS' for key in out}))
ds.to_netcdf(args.output)
