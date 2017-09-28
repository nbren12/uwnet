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

    prep_kwargs = dict(scale_input=False, scale_output=False,
                       weight_input=True, weight_output=True)

    # update with kwargs from snakemake
    prep_kwargs.update(snakemake.params.get('prep_kwargs', {}))

    # update with monkey-patched kwargs
    try:
        mod.prep_kwargs
    except NameError:
        pass
    else:
        prep_kwargs.update(mod.prep_kwargs)

    print(f"Running Cross Validation with", mod)
    print("Preparation kwargs are", prep_kwargs)

    # density
    weight = xr.open_dataarray(snakemake.input.weight)

    # Load data
    X = xr.open_dataset(snakemake.input.input)

    Y = xr.open_dataset(snakemake.input.output)

    D = xr.merge([X, Y], join='inner')

    # xarray preparation transformer
    xprep = XarrayPreparer(sample_dims=sample_dims, weight=weight, **prep_kwargs)

    inputs = list(X.data_vars)
    outputs = list(Y.data_vars)

    print("Input variables", inputs)
    print("Output variables", outputs)
    print("Sample dims", sample_dims)

    x, y = xprep.fit_transform(D[inputs], D[outputs])

    if is_mca_model:
        xprep1 = XarrayPreparer(sample_dims=sample_dims, weight=weight, **prep_kwargs)
        y_in = xprep1.fit_transform(D[outputs])
        x = [x, y_in]

    # fit model
    print(f"Fitting model")
    mod.fit(x, y)
    idx = (x.indexes['features'], y.indexes['features'])

    joblib.dump({'mod': mod, 'idx': idx}, snakemake.output.model)

    try:
        snakemake.output.prediction
    except AttributeError:
        pass
    else:
        print("Saving prediction")
        x, ytrue = xprep.transform(D[inputs], D[outputs])
        if is_mca_model:
            x = [x, None]
        ypred = mod.predict(x)

        Y = unstack_cat(xr.DataArray(ypred, ytrue.coords), 'features') \
            .unstack('samples')
        Y.to_netcdf(snakemake.output.prediction)

    # output residual
    try:
        snakemake.output.residual
    except AttributeError:
        pass
    else:
        print("Saving Residual")
        resid = unstack_cat(xr.DataArray(ytrue-ypred, ytrue.coords), 'features') \
                    .unstack('samples')
        resid.to_netcdf(snakemake.output.residual)


try:
    snakemake
except NameError:
    pass
else:
    main(snakemake)
