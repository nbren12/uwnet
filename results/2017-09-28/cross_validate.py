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

    sample_dims = snakemake.params.sample_dims

    # density
    weight = xr.open_dataarray(snakemake.input.weight)

    mod = XWrapper(snakemake.params.model, sample_dims, weight)
    print(f"Running Cross Validation with", mod._model)

    # Load data
    X_train = xr.open_dataset(snakemake.input.input_train)
    X_test = xr.open_dataset(snakemake.input.input_test)

    Y_train = xr.open_dataset(snakemake.input.output_train)
    Y_test = xr.open_dataset(snakemake.input.output_test)

    inputs = list(X_train.data_vars)
    outputs = list(Y_train.data_vars)

    D_train = xr.merge([X_train, Y_train], join='inner')
    D_test = xr.merge([X_test, Y_test], join='inner')

    ## Begin machine learning
    cv_results = []
    for params in ParameterGrid(mod._model.param_grid):
        print("Performing cross validation with the following params")
        print(params)
        cv_result = {'params': params}

        mod.set_params(**params)
        # fit model
        print(f"Fitting model")
        mod.fit(D_train[inputs], D_train[outputs])

        # compute fitting score
        cv_result['train_scores'] = mod.score(D_train[inputs], D_train[outputs])

        # compute cross validation score
        print("Computing Cross Validation Score")
        cv_result['test_scores'] = mod.score(D_test[inputs], D_test[outputs])

        # print(f"cross validation score is {score}, q1:{score_q1}, q2:{score_q2}")
        # with open(snakemake.output.r2, "w") as f:
        #     f.write(f"{score},{score_q1},{score_q2}\n")
        cv_results.append(cv_result)

    # write CV results to disk
    with open(snakemake.output.cv, "w") as f:
        json.dump(cv_results, f)

    # find the best parameter set
    best_params, best_scores = get_best_params(cv_results, score_key='total')
    print("Best scores are", best_scores)
    print("Best params are", best_params)

    # fit model using the best parameters available
    print("Re-fitting model with the best parameters")
    mod.set_params(**best_params)
    mod.fit(D_train[inputs], D_train[outputs])
    # save this model to disk
    print("Saving model to disk")
    joblib.dump(mod, snakemake.output.model)

    # compute prediction and residuals
    D = xr.concat((D_train, D_test), dim='time')
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
