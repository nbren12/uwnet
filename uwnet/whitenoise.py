import argparse
import logging

import dask.array as da
import numpy as np
import xarray as xr

import torch
from uwnet.thermo import compute_apparent_source

logger = logging.getLogger('whitenoise')


def get_error(model):
    from src.data import open_data
    data = open_data("training")
    data = data.isel(time=slice(0, 100)).compute()

    srcs = model.predict(data)

    q2 = compute_apparent_source(data.QT, data.FQT * 86400)
    q1 = compute_apparent_source(data.SLI, data.FSLI * 86400)

    return xr.Dataset({
        'QT': q2 - srcs.QT,
        'SLI': q1 - srcs.SLI
    }).dropna('time')


def cholesky_factor(C):
    return np.stack(np.linalg.cholesky(C[i]) for i in range(C.shape[0]))


class WhiteNoiseModel(object):
    """Generate random noise with correct covariance structure
    """

    def fit(self, error):

        self.time_step_ = float(error.time[1] - error.time[0])

        X = da.concatenate([error.QT.data, error.SLI.data], axis=1)

        # compute covariance
        nz = X.shape[0]
        nx = X.shape[-1]
        n = nx * nz
        C = da.einsum('tzyx,tfyx->yzf', X, X) / n
        C = C.compute()

        # shape is
        # (y, feat, feat)
        self.Q_ = cholesky_factor(C) * np.sqrt(self.time_step_)
        return self

    def __call__(self, state):
        """
        Parameters
        ----------
        state : dict

        Returns
        -------
        tend : dict
            physical tendencies
        """
        sli_key = "liquid_ice_static_energy"
        qt_key = "total_water_mixing_ratio"

        nx = state[qt_key].shape[-1]
        dt = state['dt'] / 86400
        logger.info(f"Computing white noise tendency with {dt} days")

        y = self.Q_.shape[0]
        z = self.Q_.shape[1]

        # dividing by sqrt(dt) ensures that
        # output * dt = Q sqrt{dt} N
        N = np.random.randn(y, nx, z) * np.sqrt(dt)
        W = np.einsum('yzf,yxf->zyx', self.Q_, N)

        dqt, dsli = np.split(W, 2, 0)

        # perform time step
        qt = state[qt_key] + dqt
        sli = state[sli_key] + dsli

        return {qt_key: qt, sli_key: sli}


def fit(model):
    model = torch.load(model)
    error = get_error(model)
    return WhiteNoiseModel().fit(error)
