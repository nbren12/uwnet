#!/usr/bin/env python
"""Perform partial least squares

Usage:
  pls.py -i <input_vars> -p <pred_vars> -s <sample_dims> -f <feat_dims>
         --stat FILE
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


def get_weight(stat_file):
    rho = xr.open_dataset(stat_file).RHO[-1]
    return np.sqrt(rho/integrate(rho, 'z'))


def main(input_variables, output_variables, filenames,
         sample_dims, feature_dims, stat_file,
         output):


    # open datasets
    D = xr.open_mfdataset(filenames)

    # get weight
    weight = get_weight(stat_file)

    # form data matrices
    X, x_scale = xr2mat(D[input_variables], sample_dims, feature_dims,
                        weight=weight, scale=False)
    X_mean = X.mean('samples')
    Y, y_scale = xr2mat(D[output_variables],sample_dims, feature_dims,
                        weight=weight, scale=False)
    Y_mean = Y.mean('samples')
    # compute data
    X.load()
    Y.load()
    # perform regression
    mod = PLSRegression(n_components=4, scale=False)
    mod.fit(X-X_mean, Y-Y_mean)

    pred = mod.predict(X-X_mean) + Y_mean.values
    # predicted values
    save_pls(mod, X, Y, pred, output)


def test_main():
    input_variables = ['qt', 'sl']
    output_variables = ['q1', 'q2']
    filenames = ["wd/calc/q1.nc", "wd/calc/q2.nc", "wd/calc/sl.nc", "wd/calc/qt.nc"]
    sample_dims = ('x', 'time')
    feature_dims = ('z',)

    stat_file = 'wd/stat.nc'

    main(input_variables, output_variables, filenames,
         sample_dims, feature_dims, stat_file)

if __name__ == '__main__':
    args = docopt(__doc__)
    input_vars = args['<input_vars>'].split(',')
    pred_vars = args['<pred_vars>'].split(',')
    sample_dims = args['<sample_dims>'].split(',')
    feature_dims = args['<feat_dims>'].split(',')
    output = args['-o']
    files = args['<files>']
    main(input_vars, pred_vars, files,
         sample_dims, feature_dims, args['--stat'],
         output)

