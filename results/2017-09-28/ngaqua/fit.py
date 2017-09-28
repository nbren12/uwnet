"""Perform maximum covariance analysis of q1, q2  vs sl, qt

Inputs
------
input_train, input_test
output_train, output_test

Outputs
-------
score
model

Parameters
----------
model: sklearn model object
"""
import json
import numpy as np
import xarray as xr
from xnoah.data_matrix import unstack_cat, stack_cat, compute_weighted_scale
from sklearn.externals import joblib
from sklearn.model_selection import ParameterGrid
from lib.models import *


def main(snakemake):

    sample_dims = snakemake.params.sample_dims
    is_mca_model = (snakemake.wildcards.model == 'mca')

    mod = snakemake.params.model

    return params, scores


def main(snakemake):

    sample_dims = snakemake.params.sample_dims

    # density
    weight = xr.open_dataarray(snakemake.input.weight)

    mod = XWrapper(snakemake.params.model, sample_dims, weight)
    print(f"Fitting", mod._model)

    # Load data
    X = xr.open_dataset(snakemake.input.input)
    Y = xr.open_dataset(snakemake.input.output)

    inputs = list(X.data_vars)
    outputs = list(Y.data_vars)

    D = xr.merge([X, Y])


    mod.fit(D[inputs], D[outputs])
    score = mod.score(D[inputs], D[outputs])
    print(f"Score is", score)

    joblib.dump(mod, snakemake.output.model)

    # compute prediction and residuals
    try:
        snakemake.output.prediction
    except AttributeError:
        pass
    else:
        print("Saving prediction")
        ypred = mod.predict(D[inputs])
        ypred.to_netcdf(snakemake.output.prediction)

    # output residual
    try:
        snakemake.output.residual
    except AttributeError:
        pass
    else:
        print("Saving Residual")
        resid = D[outputs]-ypred
        resid.to_netcdf(snakemake.output.residual)


try:
    snakemake
except NameError:
    pass
else:
    main(snakemake)
