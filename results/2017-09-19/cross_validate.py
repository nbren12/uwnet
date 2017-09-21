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
from models import *


def get_best_params(cv_results, score_key=0):
    best_score = -1000
    params = None
    for res in cv_results:
        score = res['test_scores'][score_key]
        if score > best_score:
            best_score = score
            params = res['params']
            scores = res['test_scores']

    return params, scores


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

    x_train, y_train = xprep.fit_transform(D_train[inputs], D_train[outputs])
    x_test, y_test = xprep.transform(D_test[inputs], D_test[outputs])

    ## Begin machine learning
    cv_results = []
    for params in ParameterGrid(mod.param_grid):
        print("Performing cross validation with the following params")
        print(params)
        cv_result = {'params': params}

        mod.set_params(**params)
        # fit model
        print(f"Fitting model")
        mod.fit(x_train, y_train)

        # compute fitting score
        y_pred = mod.predict(x_train)
        cv_result['train_scores'] = xprep.score(y_pred, y_train)

        # compute cross validation score
        print("Computing Cross Validation Score")
        y_pred = mod.predict(x_test)
        cv_result['test_scores'] = xprep.score(y_pred, y_test)

        # print(f"cross validation score is {score}, q1:{score_q1}, q2:{score_q2}")
        # with open(snakemake.output.r2, "w") as f:
        #     f.write(f"{score},{score_q1},{score_q2}\n")
        cv_results.append(cv_result)

    # write CV results to disk
    with open(snakemake.output.cv, "w") as f:
        json.dump(cv_results, f)

    # find the best parameter set
    best_params, best_scores = get_best_params(cv_results)
    print("Best scores are", best_scores)
    print("Best params are", best_params)

    # fit model using the best parameters available
    mod.set_params(**best_params)

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
