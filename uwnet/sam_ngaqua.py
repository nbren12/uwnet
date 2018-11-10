import os
import numpy as np
import torch
import logging
import attr
import xarray as xr
from toolz import valmap, curry
from uwnet.thermo import compute_apparent_source

def _dataarray_to_numpy(x):
    dim_order = ['z', 'y', 'x']
    return x.transpose(*dim_order).values


@curry
def _get_numpy_from_time(x, time):
    return x.sel(time=time, method='nearest').pipe(_dataarray_to_numpy)


class NGAquaForcing(object):
    def __init__(self, ngaqua):
        self.logger = logging.getLogger('NGAquaForcing')
        self.logger.info("Compute NGAqua Forcing from time series")

        self.sources = dict(
             FQTNN=compute_apparent_source(ngaqua.QT, ngaqua.FQT * 86400),
             FSLINN=compute_apparent_source(ngaqua.SLI, ngaqua.FSLI * 86400)
        )
        
    def get_forcings(self, time):
        self.logger.info(f"Getting forcing from day {time}")
        return valmap(_get_numpy_from_time(time=time), self.sources)


def get_ngaqua_forcing():
    path = os.environ['NGAQUA_PATH']
    ds = xr.open_dataset(path).isel(step=0).chunk({'time': 1})
    return NGAquaForcing(ds)


def save_debug(obj, state):
    path = f"{state['case']}_{state['caseid']}_{int(state['nstep']):07d}.pkl"
    print(f"Storing debugging info to {path}")
    torch.save(obj, path)

try:
    FORCING = get_ngaqua_forcing()
except:
    print("Forcing data could not be loaded")


def call_neural_network(state):
    logger = logging.getLogger(__name__)
    day = state['day']
    forcing = FORCING.get_forcings(day)
    state.update(forcing)

