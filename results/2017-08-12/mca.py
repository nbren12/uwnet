"""Perform maximum covariance analysis of q1, q2  vs sl, qt
"""
import numpy as np
from scipy.linalg import svd
import pandas as pd
import xarray as xr
from gnl.xarray import integrate


def xr2mat(fields, sample_dims, feature_dims,
           scale=True, weight=None):
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
              .transpose('samples', 'features')


def svd2xr(U, X):
    neig = U.shape[1]
    eig_idx = pd.Index(range(neig), name="m")
    Ux = xr.DataArray(U, (X.features, eig_idx))\
           .unstack('features')/weight
    return Ux


# density
rho = xr.open_dataset(snakemake.input.stat).RHO[-1]
weight = np.sqrt(rho/integrate(rho, 'z'))

# dependent variables
q1 = xr.open_dataset(snakemake.input.q1).q1
q2 = xr.open_dataset(snakemake.input.q2).q2
X = xr2mat([q1, q2], ('time', 'x'), ('z',), weight=rho)

# independent variables
sl = xr.open_dataset(snakemake.input.sl).sl
qt = xr.open_dataset(snakemake.input.qt).qt
Y = xr2mat([sl, qt], ('time', 'x'), ('z',), weight=rho)

# Perform the analysis
C = X.values.T.dot(Y.values)/(len(X.samples)-1)  # covariance matrix
U, S, Vt = svd(C, full_matrices=0)   # singular value decomposition

# Recompose data
neig = 20
Ux = svd2xr(U[:,:neig], X)
Vx = svd2xr(Vt.T[:,:neig], Y)
S = xr.DataArray(S[:neig], (np.r_[0:neig],), ('m',))

# output dataset
d = xr.concat([Ux,Vx],'variable').to_dataset("variable")
d['eig'] = S
d.to_netcdf(snakemake.output[0])
