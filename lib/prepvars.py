import os

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.externals import joblib
from lib.models import prepvar
from xnoah.data_matrix import stack_cat

def weights_to_np(w, idx):
    def f(i):
        if i < 0:
            return 1.0
        else:
            return float(w.sel(z=idx.levels[1][i]))

    return np.array([f(i) for i in idx.labels[1]])

def xarray_std_to_df(std):
    input_std = stack_cat(std, "features", ['z'])
    return pd.Series(input_std.data, index=input_std.indexes['features'], )


def weight_lrf(lrf, sig):
    sig = xarray_std_to_df(sig)
    sig_weighted_lrf = lrf.apply(lambda x: x*sig)
    return sig_weighted_lrf


def prepvars(d, input_vars, output_vars):
    return prepvar(d[input_vars]), prepvar(d[output_vars])


def get_dataset(snakemake):
    """Load data and split into training data and testing data"""
    data3d = snakemake.input.data3d
    data2d = snakemake.input.data2d

    D = xr.open_mfdataset(data3d)
    D2 = xr.open_mfdataset(data2d)

    # merge data
    D = D.merge(D2, join='inner')
    D = D.assign(Q1c=D.Q1 - D.QRAD)

    # train test split

    return D

# get variables
input_vars = snakemake.params.input_vars
output_vars = snakemake.params.output_vars

D = get_dataset(snakemake)

d_train, d_test = D.sel(time=slice(0, 50)), D.sel(time=slice(50, None))

x_train, y_train = prepvars(d_train, input_vars, output_vars)
x_test, y_test = prepvars(d_test, input_vars, output_vars)

# get input and output weights
w = xr.open_dataarray(snakemake.input.weight)
w_input = weights_to_np(w, x_train.indexes['features'])
w_output = weights_to_np(w, y_train.indexes['features'])

output_data = {
    'w': {'in': w_input, 'out': w_output},
    'train': (x_train, y_train),
    'test': (x_test, y_test)}

joblib.dump(output_data, snakemake.output[0])
