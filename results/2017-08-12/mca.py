"""Perform maximum covariance analysis of q1, q2  vs sl, qt
"""
import numpy as np
from scipy.linalg import svd
import pandas as pd
import xarray as xr
from gnl.xarray import integrate, xr2mat



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
q1 = xr.open_dataset(snakemake.input.q1).pipe(mysel)
q2 = xr.open_dataset(snakemake.input.q2).q2.pipe(mysel)

q1c = q1.q1-q1.tend
q1c.name = 'q1c'
X, scale_X = xr2mat([q1c, q2], ('time', 'x'), ('z',), weight=weight)

# independent variables
sl = xr.open_dataset(snakemake.input.sl).sl.pipe(mysel)
qt = xr.open_dataset(snakemake.input.qt).qt.pipe(mysel)
Y, scale_Y = xr2mat([sl, qt], ('time', 'x'), ('z',), weight=weight)

# Perform the analysis
C = X.values.T.dot(Y.values)/(len(X.samples)-1)  # covariance matrix
U, S, Vt = svd(C, full_matrices=0)   # singular value decomposition

# Recompose data
neig = 20
Ux = svd2xr(U[:,:neig], X, weight, scale_X)
Vx = svd2xr(Vt.T[:,:neig], Y, weight, scale_Y)
S = xr.DataArray(S[:neig]**2, (np.r_[0:neig],), ('m',))/(S**2).sum()

# rescale data
d = xr.concat([Ux,Vx],'variable') * xr.concat([scale_X, scale_Y], 'variable')
d = d/d.sel(variable='qt').max()

# output dataset
d = d.to_dataset('variable')
d['eig'] = S
d.to_netcdf(snakemake.output[0])
