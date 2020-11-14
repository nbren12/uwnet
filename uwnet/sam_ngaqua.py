"""SAM Python plugins for forcing and nudging coarse resolution SAM towards
NG-Aqua"""
import os
import torch
import logging
import xarray as xr
import numpy as np
from toolz import valmap, curry
from uwnet.thermo import compute_apparent_source
from functools import lru_cache


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
            FSLINN=compute_apparent_source(ngaqua.SLI, ngaqua.FSLI * 86400))

    def get_forcings(self, time):
        self.logger.info(f"Getting forcing from day {time}")
        return valmap(_get_numpy_from_time(time=time), self.sources)


class NGAquaNudger(object):
    """Class for nudging SAM towards NG-Aqua"""

    def __init__(self,
                 ngaqua,
                 time_scale=.125,
                 nudging_variables=('QT', 'SLI', 'U', 'V')):
        self.logger = logging.getLogger('NGAquaNudger')
        self.ngaqua = ngaqua
        self.time_scale = time_scale
        self.nudging_variables = nudging_variables

    def get_previous_time_index(self, day):
        return np.searchsorted(self.ngaqua.time.values, day, side='right') - 1

    @lru_cache(maxsize=20)
    def cached_get_numpy_at_time(self, field, time_index):
        return _dataarray_to_numpy(self.ngaqua[field].isel(time=time_index))

    def get_ngaqua_at_day(self, day):
        before = self.get_previous_time_index(day)
        after = before + 1

        t0 = float(self.ngaqua.time[before])
        t1 = float(self.ngaqua.time[after])

        dt = t1 - t0
        a = (day - t0) / dt
        assert 0 <= a < 1

        target = {}
        for key in self.nudging_variables:
            n0 = self.cached_get_numpy_at_time(key, before)
            n1 = self.cached_get_numpy_at_time(key, after)

            target[key] = (1 - a) * n0 + a * n1
        return target

    def get_nudging(self, state):
        day = state['day']
        self.logger.info(f"Nudging at day {day}")
        target = self.get_ngaqua_at_day(day)
        return self.nudge_variables(state, target)

    def nudge_variable(self, x, target):
        return (target - x) / self.time_scale

    def _get_out_key(self, key):
        return key

    def nudge_variables(self, state, target):
        output = {}
        for key in self.nudging_variables:
            out_key = self._get_out_key(key)
            src = self.nudge_variable(state[key], target[key])
            output[out_key] = src
        return output

    def __call__(self, *args):
        return self.get_nudging(*args)


def get_ngaqua_forcing():
    path = os.environ['NGAQUA_PATH']
    ds = xr.open_dataset(path).isel(step=0).chunk({'time': 1})
    return NGAquaForcing(ds)


def get_ngaqua_nudger(config):
    path = config['ngaqua']
    time_scale = config['time_scale']
    ds = xr.open_dataset(path).chunk({'time': 1})
    return NGAquaNudger(ds, time_scale=time_scale,
                        nudging_variables=['U', 'V', 'W', 'SLI', 'QT'])
