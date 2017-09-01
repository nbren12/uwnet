#!/usr/bin/env python
"""Dynamic mode decomposition for xarray data

Typically we open data in the form of a DataSet. This ensures that the dimensions are aligned.
"""
import xarray as xr
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from gnl.xarray import xr2mat


def logm(A):
    w, v = np.linalg.eig(A)

    return np.real(v@np.diag(np.log(w)) @ np.linalg.inv(v))

class DMD(object):
    def __init__(self,  n_components=3, time_dim='time', feature_dims=None,
                 input_variables=None, output_variables=None,
                 weight_variable=None,
                 scale_dim='z',
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
        self.scale_dim = scale_dim


    def prepare_input(self, X):

        if self.weight_variable is not None:
            weight = X[self.weight_variable]
            Xw = X*np.sqrt(weight)

        if self.feature_dims is None:
            feats = ['variable'] + [x for x in X.dims if x != self.time_dim]

        sample_dims = [x for x in X.dims if x not in feats]


        # scales
        if self.scale:
            scale = Xw.std([self.scale_dim] + sample_dims)
            Xw = Xw / scale

        X1 = Xw.isel(**{self.time_dim: slice(0, -1)})
        X2 = Xw.isel(**{self.time_dim: slice(1, None)})



        Xd = X1[self.input_variables]\
             .to_array()\
             .stack(features=feats, samples=sample_dims)\
             .transpose('samples', 'features')

        Yd = X2[self.output_variables]\
             .to_array()\
             .stack(features=feats, samples=sample_dims)\
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

        return P.T @ logm(At) @ P




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


def plot_coef(A):
    A1, A2 = np.split(A.T, 2, axis=0)
    A11, A12 = np.split(A1, 2, axis=1)
    A21, A22 = np.split(A2, 2, axis=1)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
    for mat, ax in zip([A11, A12, A21, A22], axs.flat):
        ax.pcolormesh(mat)

    plt.show()


def main():
    # D = xr.open_dataset("wd/A64/3d/QV.nc").isel(x=0)
    sl = xr.open_dataset("wd/calc/sl.nc").isel(x=0)
    qt = xr.open_dataset("wd/calc/qt.nc").isel(x=0)
    stats = xr.open_dataset("wd/stat.nc")
    rho = stats.RHO[-1]

    # compute weight
    D = xr.merge((sl, qt, rho))


    dmd = DMD(time_dim='time', n_components=9,
              input_variables=['qt', 'sl'],
              weight_variable='RHO')
    dmd.fit(D)

    y =  dmd.predict(D)


    A = dmd.generator()

    plot_coef(A)
    from IPython import embed; embed()

if __name__ == '__main__':
    main()
