"""Perform maximum covariance analysis of q1, q2  vs sl, qt
"""
import numpy as np
from scipy.linalg import svd
import pandas as pd
import xarray as xr
from gnl.xarray import integrate, xr2mat
from gnl.data_matrix import NormalizedDataMatrix


# density
weight = xr.open_dataarray(snakemake.input.weight)

# dependent variables
input = xr.open_dataset(snakemake.input.X)
output = xr.open_dataset(snakemake.input.Y)

D = xr.merge([input, output], join='inner')


common_kwargs = dict(feature_dims=['z'], sample_dims=['time', 'x'], weight=weight, apply_weight=True)

in_dm = NormalizedDataMatrix(variables=['sl', 'qt', 'LHF', 'SHF'],
                             **common_kwargs)
out_dm = NormalizedDataMatrix(variables=['q1', 'q2'],
                              **common_kwargs)

# might want to generate cross validation here

X = in_dm.transform(D)
Y = out_dm.transform(D)

np.savez(snakemake.output.X, X)
np.savez(snakemake.output.Y, Y)

