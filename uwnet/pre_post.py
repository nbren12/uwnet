"""Tools for pre/post processing inputs/outputs from neural networks

Unlike pytorch's builtin tools, this code allows building pytorch modules from
scikit-learn estimators.
"""
import pandas as pd
import numpy as np
import xarray as xr
from torch import nn

from uwnet.thermo import compute_apparent_source
from uwnet.modules import MapByKey, LinearFixed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def prepare_data(data, exog=['QT', 'SLI', 'SST', 'SOLIN']):
    """Flatten XArray dataset into dataframe.

    The columns of the dataframe are the variables names and height indices.
    The rows are the samples, which are taken along 'x', 'y', and 'time'.
    """
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
    keys = ['QT', 'SLI', 'SHF', 'LHF', 'SOLIN', 'SST']
    df = prepare_data(data, exog=keys)
    transformers = {}
    for key in keys:
        x = df[key]
        n = x.shape[1]

        if n == 1:
            transformers[key] = fit_scaler_transform(x)
        else:
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


def get_pre_post_orig(dataset, n):
    from .normalization import get_mean_scale, Scaler
    inputs = [('QT', n), ('SLI', n),
              ('SST', 1), ('SOLIN', 1)]
    mean, scale  = get_mean_scale(dataset)
    scaler = Scaler(mean, scale)
    scaler.outputs = inputs
    scaler.inputs = inputs
    # post processor
    outputs = (('QT', n), ('SLI', n))
    post = Post(scale['QT'], outputs)
    return scaler, post


def get_pre_post(kind, data, *args):
    if kind == 'pca':
        return get_pre(data, m=10), get_post(data, m=10)
    elif kind == 'orig':
        return get_pre_post_orig(data, *args)
    elif kind == 'mix':
        pre = get_pre(data, m=20)
        _ , post= get_pre_post_orig(data, *args)
        return pre, post
