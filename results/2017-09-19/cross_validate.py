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
import xarray as xr
from gnl.data_matrix import NormalizedDataMatrix
from sklearn.externals import joblib


def main(snakemake):

    # density
    weight = xr.open_dataarray(snakemake.input.weight)

    # Load data
    X_train = xr.open_dataset(snakemake.input.input_train)
    X_test = xr.open_dataset(snakemake.input.input_test)

    Y_train = xr.open_dataset(snakemake.input.output_train)
    Y_test = xr.open_dataset(snakemake.input.output_test)

    D_train = xr.merge([X_train, Y_train], join='inner')
    D_test = xr.merge([X_test, Y_test], join='inner')

    common_kwargs = dict(feature_dims=['z'], sample_dims=['time', 'x'],
                         weight=weight, apply_weight=True)

    # normalized tranformer matrices
    in_dm = NormalizedDataMatrix(variables=['sl', 'qt', 'LHF', 'SHF'],
                                 **common_kwargs)
    out_dm = NormalizedDataMatrix(variables=['q1', 'q2'],
                                  **common_kwargs)

    # Create 2D data matrices
    x_train = in_dm.transform(D_train)
    y_train = out_dm.transform(D_train)

    x_test = in_dm.transform(D_test)
    y_test = out_dm.transform(D_test)

    # mod is in snakemake
    mod = snakemake.params.model
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
        x = in_dm.transform(D)
        # need to run this to initialize DM
        ytrue = out_dm.transform(D)
        ypred = mod.predict(x)

        Y = out_dm.inverse_transform(ypred)
        Y.to_netcdf(snakemake.output.prediction)

    # output residual
    try:
        snakemake.output.residual
    except AttributeError:
        pass
    else:
        print("Saving Residual")
        resid = out_dm.inverse_transform(ytrue-ypred)
        resid.to_netcdf(snakemake.output.residual)


try:
    snakemake
except NameError:
    pass
else:
    main(snakemake)
