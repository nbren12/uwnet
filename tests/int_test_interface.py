import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from toolz import merge_with, take_nth
from tqdm import tqdm

import xarray as xr
from uwnet.interface import step_model


def load_model(path):
    import uwnet.model
    import torch
    d = torch.load(path)['dict']
    return uwnet.model.MLP.from_dict(d)


def scale_data(ds):
    scales = {'qt': 1e-3, 'FQT': 1e-3 / 86400, 'FSL': 1 / 86400, 'time': 86400}
    for key in scales:
        ds[key] *= scales[key]
    return ds


def load_data(path):
    idx = dict(x=0, y=32)
    ds = xr.open_zarr(path)\
           .isel(idx)\
           .pipe(scale_data)
    ds['time'] -= ds['time'][0]

    x = {key: val.values for key, val in ds.items()}

    nt, nz = x['qt'].shape

    time = ds['time'].values

    for key, val in x.items():
        if val.size == nt:
            val.shape = [nt, 1, 1, 1]

        if val.ndim > 0 and val.shape[0] == nt:
            val.shape = (nt, val.shape[-1], 1, 1)
            x[key] = interp1d(time, val, axis=0)

    return x


def run(model, x, t0, n, dt):

    qt = x['qt'](t0)
    sl = x['sl'](t0)

    t = t0
    for i in tqdm(range(n)):
        out = step_model(model.step, dt, x['layer_mass'], qt, sl, x['FQT'](t),
                         x['FSL'](t), x['U'](t), x['V'](t), x['SST'](t),
                         x['SOLIN'](t))
        qt = out['qt']
        sl = out['sl']

        t += dt
        out['time'] = t
        yield out


dt = 30  # seconds
day = 86400
nt = 40 * day // dt
nout = int(3600 // dt)

model_path = "./10_test_db/4.pkl"
data_path = "./all.1.zarr"

model = load_model(model_path)
x = load_data(data_path)

output = merge_with(np.stack, take_nth(nout, run(model, x, 0.0, nt, dt=dt)))

plt.contourf(output['time'], x['p'], output['qt'].squeeze().T)
plt.ylim([1000, 10])
plt.colorbar()
plt.show()
