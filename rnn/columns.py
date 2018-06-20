import argparse
import xarray as xr

import torch
import yaml
from toolz import *

from . import model, utils
from .prepare_data import get_dataset


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
    return x.isel(x=slice(0,1), y=slice(32,33))

paths = config['paths']
data = get_dataset(paths, post=post)


# prepare input for mod
data = {key: torch.tensor(val).unsqueeze(0) for key, val in data[0].items()}
init_cond = {key: data[key][:, 0] for key in mod.progs}

orig = {}
for key in mod.progs | set(['p', 'layer_mass']):
    orig[key] = data.pop(key)

# finally run model
with torch.no_grad():
    out = mod(init_cond, data)

# save data to netcdf
out_da = {key: xr.DataArray(val.detach().numpy(), dims=['b', 'time', 'z']) for key, val in out.items()}

ds = xr.Dataset(out_da)
ds.to_netcdf("out.nc")

# plot
import matplotlib.pyplot as plt
plt.subplot(211)
plt.contourf(out['qt'].numpy()[0,:].T)
plt.colorbar()
plt.subplot(212)
plt.contourf(orig['qt'].numpy()[0,:].T)
plt.colorbar()
plt.show()
