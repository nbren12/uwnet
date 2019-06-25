"""Tools for pre/post processing inputs/outputs from neural networks

Unlike pytorch's builtin tools, this code allows building pytorch modules from
scikit-learn estimators.
"""
import pandas as pd
import numpy as np
import xarray as xr
import torch
from torch import nn

from uwnet.thermo import compute_apparent_source
from uwnet.modules import MapByKey, LinearFixed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import logging

logger = logging.getLogger(__name__)


def prepare_data(data, exog=['QT', 'SLI', 'SST', 'SOLIN'], sample=None):
    """Flatten XArray dataset into dataframe.

    The columns of the dataframe are the variables names and height indices.
    The rows are the samples, which are taken along 'x', 'y', and 'time'.
    """
    logger.info("Flattening xarray Dataset to pandas DataFrame")
    size = data[exog].nbytes/1e9
    logger.info(f"Size: {size} GB")
    vals = []
    names = []
    sample_dims = ['x', 'y', 'time']
    for key in exog:
        val = data[key].stack(s=sample_dims)
        if 'z' not in val.dims:
            val = val.expand_dims('z')

        arr = val.transpose('s', 'z').values
        vals.append(arr)

        for z in range(arr.shape[1]):
            names.append((key, z))

    inputs = np.concatenate(vals, axis=1)
    idx = pd.MultiIndex.from_tuples(names)
    return pd.DataFrame(inputs, columns=idx)


def fit_pca_inverse_transform(q1, n=16):
    pca = PCA(n_components=n, whiten=True)
    pca.fit(q1)
    return LinearFixed.from_affine(pca.inverse_transform, n)


def fit_pca_transform(x, n=16):
    pca = PCA(n_components=n, whiten=True)
    pca.fit(x)
    return LinearFixed.from_affine(pca.transform, x.shape[1])


def fit_scaler_transform(x):
    scaler = StandardScaler().fit(x)
    return LinearFixed.from_affine(scaler.transform, x.shape[1])


def get_post(data, m=20):
    logger.info("Fitting PCA models for outputs")
    q1 = compute_apparent_source(data.SLI, data.FSLI * 86400).dropna('time')
    q2 = compute_apparent_source(data.QT, data.FQT * 86400).dropna('time')

    ds = xr.Dataset({'Q1': q1, 'Q2': q2})
    df = prepare_data(ds, exog=['Q1', 'Q2'])

    funcs = {
        'SLI': fit_pca_inverse_transform(df['Q1'], m),
        'QT': fit_pca_inverse_transform(df['Q2'], m),
    }

    return MapByKey(funcs)


def get_pre(data, m=20):
    logger.info("Building preprocessing module")
    keys = ['QT', 'SLI', 'SHF', 'LHF', 'SOLIN', 'SST']
    df = prepare_data(data, exog=keys)
    transformers = {}
    for key in keys:
        x = df[key]
        n = x.shape[1]

        if n == 1:
            logger.info(f"Fitting Scaler for {key}")
            transformers[key] = fit_scaler_transform(x)
        else:
            logger.info(f"Fitting PCA for {key}")
            transformers[key] = fit_pca_transform(x, m)

    return MapByKey(transformers)



class Post(nn.Module):

    def __init__(self, q0, inputs):
        super(Post, self).__init__()
        self.register_buffer('q0', q0)
        self.inputs = inputs

    def forward(self, d):
        q0 = self.q0.clamp(max=1)
        d['QT'] = d['QT'] * q0
        return d


class LowerAtmosInput(nn.Module):

    @property
    def outputs(self):
        return [
            ('QT', 15),
            ('SLI', 18),
            ('SST', 1),
            ('SOLIN', 1),
        ]

    def forward(self, x):
        out = {}
        for key, n in self.outputs:
            out[key] = x[key][..., :n]
        return out


class IdentityOutput(nn.Module):
    inputs = [
        ('QT', 34),
        ('SLI', 34),
    ]

    outputs = [
        ('QT', 34),
        ('SLI', 34),
    ]

    def forward(self, x):
        return x


class Sequential(nn.Sequential):
    @property
    def inputs(self):
        return self[0].inputs

    @property
    def outputs(self):
        return self[-1].outputs


def get_pre_post_orig(data, data_loader, n):
    from .normalization import Scaler
    inputs = [('QT', n), ('SLI', n),
              ('SST', 1), ('SOLIN', 1)]
    scaler = Scaler().fit_generator(data_loader)
    scaler.outputs = inputs
    scaler.inputs = inputs
    # post processor
    outputs = (('QT', n), ('SLI', n))
    post = Post(scaler.scale['QT'], outputs)
    return scaler, post


def get_pre_post(data, data_loader, _config):
    kind = _config['kind']
    logger.info(f"Getting pre/post processor of type {kind}")

    if kind == 'pca':
        return get_pre(data, m=20), get_post(data, m=20)
    elif kind == 'saved':
        path = _config['path']
        logger.info(f"Loading pre/post module from {path}")
        return torch.load(path)
    elif kind == 'lower_atmos':
        pre, post = get_pre_post_orig(data, data_loader, n=34)
        lower = LowerAtmosInput()
        pre = Sequential(pre, lower)
        return pre, post
    elif kind == 'orig':
        return get_pre_post_orig(data, data_loader, n=34)
    elif kind == 'identity':
        return LowerAtmosInput(), IdentityOutput()
    else:
        raise NotImplementedError
