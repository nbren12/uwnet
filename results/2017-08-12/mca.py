"""Perform maximum covariance analysis of q1, q2  vs sl, qt
"""
import numpy as np
from scipy.linalg import svd
import pandas as pd
import xarray as xr
from gnl.xarray import integrate


def xr2mat(fields, sample_dims, feature_dims,
           scale=True, weight=1.0):
    """Prepare list of data arrays for input to Machine Learning

    Parameters
    ----------
    fields: Dataset object
        input dataset
    sample_dims: tuple of string
        Dimensions which will be considered samples
    feature_dims: tuple of strings
        dimensions which be considered features
    scale: Bool
        center and scale the output. [default: True]
    weight: xarray
        weight the xarray fields using this object. Typically,
        np.sqrt(rho)/integrate(rho, 'z') for mass aware methods.
    """
    normalize_dim='z'   # this could be an argument

    fields = xr.merge(fields)
    dat = fields.to_array() * weight

    if scale:
        mu = dat.mean(sample_dims)
        V = np.sqrt(integrate((dat-mu)**2, normalize_dim)).mean(sample_dims)
        dat = (dat-mu)/V

    return dat.stack(features=('variable',)+feature_dims, samples=sample_dims)\
              .transpose('samples', 'features'), V


def svd2xr(U, X, weight, scale):
    neig = U.shape[1]
    eig_idx = pd.Index(range(neig), name="m")
    Ux = xr.DataArray(U, (X.features, eig_idx))\
           .unstack('features')*scale/weight
    return Ux

def mysel(x):
    return x.sel(time=slice(20,None))

# density
rho = xr.open_dataset(snakemake.input.stat).RHO[-1]
weight = np.sqrt(rho/integrate(rho, 'z'))

# dependent variables
q1 = xr.open_dataset(snakemake.input.q1).q1.pipe(mysel)
q2 = xr.open_dataset(snakemake.input.q2).q2.pipe(mysel)
X, scale_X = xr2mat([q1, q2], ('time', 'x'), ('z',), weight=weight, scale=False)

# independent variables
sl = xr.open_dataset(snakemake.input.sl).sl.pipe(mysel)
qt = xr.open_dataset(snakemake.input.qt).qt.pipe(mysel)
Y, scale_Y = xr2mat([sl, qt], ('time', 'x'), ('z',), weight=weight, scale=False)

from IPython import embed; embed()
