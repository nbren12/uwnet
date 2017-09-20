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
import models


def get_input_output(ds, weight,
                     inputs=['sl', 'qt', 'LHF', 'SHF'],
                     outputs=['q1', 'q2']):

    sample_dims = ['x', 'time']

    # normalize inputs
    scales = compute_weighted_scale(weight, sample_dims, ds[inputs])
    in_ds = ds[inputs]
    # in_ds = (in_ds-in_ds.mean(axis=0))/scales*np.sqrt(weight)
    # in_ds = in_ds * np.sqrt(weight)

    # do not normalize outputs
    # weight them though...
    out_ds = ds[outputs]


    return models.get_mat(in_ds), models.get_mat(out_ds)



def main(snakemake):

    inputs = ['qt', 'sl', 'LHF', 'SHF']
    # density
    weight = xr.open_dataarray(snakemake.input.weight)

    # Load data
    X_train = xr.open_dataset(snakemake.input.input_train)
    X_test = xr.open_dataset(snakemake.input.input_test)

    Y_train = xr.open_dataset(snakemake.input.output_train)
    Y_test = xr.open_dataset(snakemake.input.output_test)

    D_train = xr.merge([X_train, Y_train], join='inner')
    D_test = xr.merge([X_test, Y_test], join='inner')

    # Create 2D data matrices
    x_train, y_train = get_input_output(D_train, weight)
    x_test, y_test = get_input_output(D_test, weight)

    # mod is in snakemake
    mod = models.RidgeRemover
    # mod = snakemake.params.model
    print(f"Fitting {mod}")
    mod.fit(x_train, y_train)

    score = mod.score(x_test, y_test)
    print(f"Cross validation score is {score}")

    # write score to file
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
        x, ytrue = get_input_output(D, weight)
        ypred = mod.predict(x)

        Y = unstack_cat(xr.DataArray(ypred, ytrue.coords), 'features')\
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
