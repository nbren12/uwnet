"""Debias a torch network using LASSO"""
import argparse

import numpy as np
import torch
import xarray as xr
from sklearn.linear_model import Lasso

from uwnet.thermo import compute_apparent_source
from uwnet.xarray_interface import XarrayWrapper
from uwnet.numpy_interface import NumpyWrapper


class LassoDebiasedModel(object):
    """Debias a model using LASSO

    Parameters
    ----------
    model : callable
    mapping : seq of tuples
        sequence of (prog_var, model_output_var, target_var) tuples. These
        are used to define the pointwise regression models.
    """

    def __init__(self, model, mapping):
        "docstring"
        self.model = model
        self.coefs_ = {}
        self.mapping = mapping

    @property
    def xmodel(self):
        return XarrayWrapper(self.model)

    @property
    def npmodel(self):
        return NumpyWrapper(self.model)

    def fit(self, input):
        """

        Parameters
        ----------
        input : xr.Dataset
            input variables
        """
        output = self.xmodel(input)

        for args in self.mapping:
            print(f"Fitting regression model for {args}")
            prog, pred, target = args
            data = get_regression_problem(input, output, prog, pred, target)
            # These coefficients have shape (3, nz, ny)
            # reshaping to (3, nz, ny, 1) makes the broadcastable
            coefs = get_lasso_coefficients(data)
            self.coefs_[(prog, pred)] = coefs
        return self

    def predict(self, input):
        """
        Parameters
        ----------
        input : xr.Datset
        """
        output = XarrayWrapper(self.model)(input)

        for (prog, pred), coefs in self.coefs_.items():
            debiased = (
                input[prog] * coefs[0] + coefs[1] * output[pred] + coefs[2])
            output[pred] = debiased
        return output

    @property
    def numpy_coefs_(self):
        """Coefficients of LASSO
        """
        out = {}
        required_dim_order = ['dim_0', 'z', 'y']
        for key, coefs in self.coefs_.items():
            coefs = coefs.transpose(*required_dim_order)
            out[key] = coefs.values[..., np.newaxis]
        return out

    def predict_with_numpy(self, input):
        output = self.npmodel(input)

        for (prog, pred), coefs in self.numpy_coefs_.items():
            debiased = (
                input[prog] * coefs[0] + coefs[1] * output[pred] + coefs[2])
            output[pred] = debiased
        return output

    def __call__(self, input):
        return self.predict_with_numpy(input)


def insert_apparent_sources(ds, prognostics):
    sec_to_day = 86400
    for prog in prognostics:
        src = 'F' + prog
        target_key = 'Q' + prog
        target = compute_apparent_source(ds[prog], ds[src] * sec_to_day)
        ds = ds.assign(**{target_key: target})

    return ds


def get_regression_problem(input, output, prog, pred, target, time_dim='time'):

    regression_data = xr.Dataset({
        'prog': input[prog],
        'target': input[target],
        'pred': output[prog]
    })

    return regression_data.dropna(time_dim)


def get_lasso_coefficients(regression_data):
    out_dims = ['y', 'z']

    # input vars
    input_vars = ['prog', 'pred']
    output_var = 'target'

    def fit_linear(dfjk):
        df = dfjk.to_dataframe().reset_index()
        x_train = df[input_vars]
        y_train = df[output_var]

        debiaser = Lasso(alpha=.0001, normalize=True).fit(x_train, y_train)
        params = np.append(debiaser.coef_, debiaser.intercept_)
        return xr.DataArray(params)

    groupby = regression_data.stack(g=out_dims).groupby('g')
    params = groupby.apply(fit_linear)
    params = params.unstack('g')
    return params
