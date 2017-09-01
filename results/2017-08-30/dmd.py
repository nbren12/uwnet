#!/usr/bin/env python
"""Dynamic mode decomposition for xarray data

Typically we open data in the form of a DataSet. This ensures that the dimensions are aligned.
"""
import xarray as xr
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from gnl.xarray import xr2mat


class DMD(object):
    def __init__(self,  n_components=3, time_dim='time', feature_dims=None,
                 input_variables=None, output_variables=None,
                 weight_variable=None,
                 scale=True):
        self.n_components = n_components
        self.time_dim = time_dim
        self.feature_dims = feature_dims
        self.input_variables = input_variables

        if output_variables is None:
            self.output_variables = input_variables
        else:
            self.output_variables = output_variables

        self.weight_variable = weight_variable
        self.scale = scale


    def prepare_input(self, X):

        if self.weight_variable is not None:
            weight = X[self.weight_variable]
            Xw = X*np.sqrt(weight)

        X1 = Xw.isel(**{self.time_dim: slice(0, -1)})
        X2 = Xw.isel(**{self.time_dim: slice(1, None)})

        if self.feature_dims is None:
            feats = [x for x in X.dims if x != self.time_dim]\
                + ['variable']

        samples = [x for x in X.dims if x not in feats]

        Xd = X1[self.input_variables]\
             .to_array()\
             .stack(features=feats, samples=samples)\
             .transpose('samples', 'features')

        Yd = X2[self.output_variables]\
             .to_array()\
             .stack(features=feats, samples=samples)\
             .transpose('samples', 'features')

        return Xd, Yd



    def fit(self, X):
        Xd, Yd = self.prepare_input(X)

        self.x_mean_ = Xd.mean('samples')
        self.y_mean_ = Yd.mean('samples')


        self.svd_ = TruncatedSVD(self.n_components)

        xa = self.svd_.fit_transform(Xd-self.x_mean_)
        ya = self.svd_.transform(Yd-self.y_mean_)
        self.lin_ = LinearRegression(fit_intercept=False, normalize=False)
        self.lin_.fit(xa, ya)

        P  = self.svd_.components_
        self.coef_ = P.T @ self.lin_.coef_ @ P


    def generator(self):
        P = self.svd_.components_
        At = self.lin_.coef_

        w, v = np.linalg.eig(At)

        lam = np.diag(np.log(w))
        Bt = np.real(v @ lam @ np.linalg.inv(v))

        return P.T @ Bt @ P




    def predict(self, X):
        Xd, Yd = self.prepare_input(X)

        xa = self.svd_.transform(Xd - self.x_mean_)
        ya = self.svd_.inverse_transform(self.lin_.predict(xa)) 
        return xr.DataArray(ya, Yd.coords) + self.y_mean_


def test_DMD():
    air = xr.tutorial.load_dataset("air_temperature")

    # compute weight
    air['weight'] = np.cos(air.lat/360 * 2 * np.pi)


    dmd = DMD(time_dim='time',
              input_variables=['air'],
              weight_variable='weight')
    dmd.fit(air)
    import matplotlib.pyplot as plt
    eig0 = xr.DataArray(dmd.coef_[:,1000], dmd.y_mean_.coords).unstack('features')
    eig0.plot()
    plt.show()
    assert False

def main():
    # D = xr.open_dataset("wd/A64/3d/QV.nc").isel(x=0)
    D = xr.open_dataset("wd/A64/3d/TABS.nc").isel(x=0)
    stats = xr.open_dataset("wd/stat.nc")
    rho = stats.RHO[-1]

    # compute weight
    D = xr.merge((D, rho))


    dmd = DMD(time_dim='time', n_components=4,
              input_variables=['TABS'],
              weight_variable='RHO')
    dmd.fit(D)

    y =  dmd.predict(D)

    import matplotlib.pyplot as plt

    A = dmd.generator()
    plt.pcolormesh(A)
    plt.show()

if __name__ == '__main__':
    main()
