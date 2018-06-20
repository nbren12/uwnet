import argparse

# plot
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import xarray as xr

from . import model
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
    return x.isel(x=slice(0, 1), y=slice(50, 51)).sortby('time')
    # return x.isel(x=slice(0, 1), y=slice(32, 33)).sortby('time')


paths = config['paths']
data = get_dataset(paths, post=post)

# prepare input for mod
batch = {key: torch.tensor(val).unsqueeze(0) for key, val in data[0].items()}

# finally run model
with torch.no_grad():
    out = mod(batch, n=1)

# save batch to netcdf
out_da = {
    key: xr.DataArray(val.detach().numpy(), dims=['b', 'time', 'z'])
    for key, val in out.items()
}

ds = xr.Dataset(out_da)
ds.to_netcdf("out.nc")

kwargs = dict(levels=np.r_[0:11] * 2)
plt.subplot(211)
plt.contourf(out['qt'].numpy()[0, :].T, **kwargs)
plt.colorbar()
plt.subplot(212)
z = data.data.qt.squeeze().values
plt.contourf(z.T, **kwargs)
plt.colorbar()
plt.show()
