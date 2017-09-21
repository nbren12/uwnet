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
import numpy as np
import xarray as xr
from xnoah.data_matrix import unstack_cat, stack_cat, compute_weighted_scale
from sklearn.externals import joblib
from models import *


def _unstack(y, coords):
    return unstack_cat(xr.DataArray(y, coords), 'features') \
             .unstack('samples')


def score_dataset(y_true, y, sample_dims, return_scalar=True):

    if not set(y.dims)  >= set(sample_dims):
        raise ValueError("Sample dims must be a subset of data dimensions")

    # means
    ss = ((y_true - y_true.mean(sample_dims))**2).sum(sample_dims).sum()

    # prediction
    sse = ((y_true - y)**2).sum(sample_dims).sum()

    r2 = 1- sse/ss
    return r2



def main(snakemake):

    inputs = ['LHF', 'SHF', 'qt', 'sl']
    outputs = ['q1', 'q2']
    sample_dims = ['x', 'time']

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
    X_train = xr.open_dataset(snakemake.input.input_train)
    X_test = xr.open_dataset(snakemake.input.input_test)

    Y_train = xr.open_dataset(snakemake.input.output_train)
    Y_test = xr.open_dataset(snakemake.input.output_test)

    D_train = xr.merge([X_train, Y_train], join='inner')
    D_test = xr.merge([X_test, Y_test], join='inner')

    # xarray preparation transformer
    xprep = XarrayPreparer(sample_dims=sample_dims, weight=weight, **prep_kwargs)

    # fit model
    print(f"Fitting model")
    x_train, y_train = xprep.fit_transform(D_train[inputs], D_train[outputs])
    mod.fit(x_train, y_train)

    # compute cross validation score
    print("Computing Cross Validation Score")
    x_test, y_test = xprep.transform(D_test[inputs], D_test[outputs])
    y_pred = mod.predict(x_test)

    # turn into Dataset
    y_true_dataset = _unstack(y_test, y_test.coords)
    y_pred_dataset = _unstack(y_pred, y_test.coords)


    # This might be unnessary
    # I think this might just be removed in the R2 calculation
    # also weighting would be removed
    if prep_kwargs['scale_output']:
        y_true_dataset *= xprep.scale_y_
        y_pred_dataset *= xprep.scale_y_



    r2 = score_dataset(y_true_dataset, y_pred_dataset, sample_dims)
    print(f"cross validation score is {r2}")
    with open(snakemake.output.r2, "w") as f:
        for key, val in r2.items():
            val = float(val)
            f.write(f"{key},{val}\n")

    # write fitted mod to file
    joblib.dump(mod, snakemake.output.model)

    # compute prediction and residuals
    D = xr.concat((D_train, D_test), dim='time')
    try:
        snakemake.output.prediction
    except AttributeError:
        pass
    else:
        print("Saving prediction")
        x, ytrue = xprep.transform(D[inputs], D[outputs])
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
