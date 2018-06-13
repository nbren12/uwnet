import numpy as np
import xarray as xr
import torch
from lib.torch import column_run, ForcedStepper, TrainingData
import torch
import logging
from toolz.curried import valmap
from toolz import first
logging.basicConfig(level=logging.DEBUG)

i = snakemake.input

state = i['state']
nsteps = snakemake.params.get('nsteps', 1)

files = [
    (i.tend, ('FQT', 'FSL')),
    (i.cent, ('QV', 'TABS', 'QN', 'QP', 'QRAD')),
    (i.stat, ('p', 'RHO')),
    (i['2d'], ('LHF', 'SHF', 'SOLIN')),
]

data = TrainingData.from_var_files(files)
nt, ny, nx, nz = data.FQT.shape

loader = data.get_loader(nt, batch_size=ny * nx * nz, shuffle=False)
input_data = first(loader)

model = ForcedStepper.from_file(state)
model.eval()

model.nsteps = 1
print("nsteps", nsteps)
with torch.no_grad():
    out = model(input_data)


def unstackdiag(x):
    shape = (nt - 1, ny, nx)
    return x.data.numpy().reshape(shape)


def unstack2d(x):
    shape = (nt, ny, nx)
    return x.data.numpy().reshape(shape)


def unstack3d(x):
    shape = (nt, ny, nx, nz)
    return x.data.numpy().reshape(shape)


output = {
    'qt': (['time', 'y', 'x', 'z'], unstack3d(out['prognostic']['qt'])[1:]),
    'sl': (['time', 'y', 'x', 'z'], unstack3d(out['prognostic']['sl'])[1:]),
}

for key, val in out['diagnostic'].items():
    output[key] = (['time', 'y', 'x'], unstackdiag(val))

output['p'] = (['z'], data.p)

coords = {'z': data.z, 'x': data.x, 'y': data.y}

xr.Dataset(output, coords=coords).to_netcdf(snakemake.output[0])
