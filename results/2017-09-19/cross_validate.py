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


def main(snakemake):

    inputs = ['LHF', 'SHF', 'qt', 'sl']
    outputs = ['q1', 'q2']

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
    xprep = XarrayPreparer(sample_dims=['x','time'], weight=weight)

    # regression pipeline
    mod = make_pipeline(NullFeatureRemover(), Ridge(1.0, normalize=True))

    # fit model
    print(f"Fitting {mod}")
    x_train, y_train = xprep.fit_transform(D_train[inputs], D_train[outputs])
    feats=mod.fit(x_train, y_train)

    # compute cross validation score
    x_test, y_test = xprep.transform(D_test[inputs], D_test[outputs])
    score = mod.score(x_test, y_test)
    print(f"Cross validation score is {score}")

    # save score to file
    with open(snakemake.output.score, "w") as f:
        f.write(f"{score}")


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
