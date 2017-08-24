#!/usr/bin/env python
"""Perform partial least squares

Usage:
  pls.py -i <input_vars> -p <pred_vars> -s <sample_dims> -f <feat_dims>
         -w <weight>
         -o <output> <files>...

Example:
  
"""
import numpy as np
import pandas as pd
import xarray as xr
from docopt import docopt
from sklearn.cross_decomposition import PLSRegression

from gnl.xarray import integrate, xr2mat


def flatten_output(x):

    for dim in ['features', 'samples']:
        try:
            x = x.unstack(dim)
        except ValueError:
            pass
    return x.to_dataset(dim='variable')


def parse_pls_output(mod, X, Y, pred):
    ncomp = mod.n_components
    m = pd.Index(range(ncomp), name='m')
    # get weights
    xw = xr.DataArray(mod.x_weights_, (X.coords['features'], m), name="xw")
    yw = xr.DataArray(mod.y_weights_, (Y.coords['features'], m), name="yw")
    # get prediction
    y_pred = xr.DataArray(pred, Y.coords, name="pred")
    return xw, yw, y_pred


def save_pls(mod, X, Y, pred, output_name):
    xw, yw, y_pred = parse_pls_output(mod, X, Y, pred)
    xw.pipe(flatten_output).to_netcdf(output_name, group="x_weights")
    yw.pipe(flatten_output).to_netcdf(output_name, group="y_weights", mode="a")
    y_pred.pipe(flatten_output).to_netcdf(output_name, group="pred", mode="a")

    np.savez('calc/lrf.npz', mod.coef_)



def main(input_variables, output_variables, filenames,
         sample_dims, feature_dims, weight_file, output):


    # open datasets
    D = xr.open_mfdataset(filenames)

    # get weight
    weight = xr.open_dataarray(weight_file)
    Dw = D * weight

    # form data matrices
    X, x_scale = xr2mat(Dw[input_variables], sample_dims, feature_dims,
                        scale=False)
    Y, y_scale = xr2mat(Dw[output_variables],sample_dims, feature_dims,
                        scale=False)
    # compute data
    X.load()
    Y.load()
    # perform regression
    mod = PLSRegression(n_components=4, scale=False)
    mod.fit(X, Y)

    pred = mod.predict(X)
    # predicted values
    save_pls(mod, X, Y, weight, pred, output)


def test_main():
    input_variables = ['qt', 'sl']
    output_variables = ['q1', 'q2']
    filenames = ["wd/calc/q1.nc", "wd/calc/q2.nc", "wd/calc/sl.nc", "wd/calc/qt.nc"]
    sample_dims = ('x', 'time')
    feature_dims = ('z',)


    main(input_variables, output_variables, filenames,
         sample_dims, feature_dims, weight_file)

try:
    input_vars = snakemake.params.input_vars
    pred_vars = snakemake.params.pred_vars
    feature_dims = snakemake.params.feature_dims
    sample_dims = snakemake.params.sample_dims

    files = snakemake.input.files
    output = snakemake.output[0]

    main(input_vars, pred_vars, files,
         sample_dims, feature_dims,
         snakemake.input.weight, output)

except NameError:

    if __name__ == '__main__':
        args = docopt(__doc__)
        input_vars = args['<input_vars>'].split(',')
        pred_vars = args['<pred_vars>'].split(',')
        sample_dims = args['<sample_dims>'].split(',')
        feature_dims = args['<feat_dims>'].split(',')
        output = args['-o']
        files = args['<files>']
        main(input_vars, pred_vars, files,
            sample_dims, feature_dims, args['-w'],
            output)

