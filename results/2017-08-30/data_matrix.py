"""Module for converting xarray datasets to and from matrix formats for Machine
learning purposes.

"""
import numpy as np
import xarray as xr


class DataMatrix(object):
    """Matrix for inputting/outputting datamatrices from xarray dataset objects

    """

    def __init__(self, feature_dims, sample_dims, variables):
        self.feature_dims = feature_dims
        self.sample_dims = sample_dims
        self.dims = {'samples': sample_dims, 'features': feature_dims}
        self.variables = variables

    def dataset_to_mat(self, X):
        Xs = X.stack(samples=self.sample_dims, features=self.feature_dims)\
              .transpose('samples', 'features')

        self._var_coords = {k: Xs[k].coords for k in self.variables}

        # store offsets
        offset = 0
        self._var_slices = {}
        one_sample = Xs.isel(samples=0)
        for k in self.variables:
            nfeat = one_sample[k].size
            self._var_slices[k] = slice(offset, offset+nfeat)
            offset += nfeat

        return np.hstack(self.column_var(Xs[k]) for k in self.variables)

    def mat_to_dataset(self, X):

        data_dict = {}

        for k in self.variables:
            data = X[:, self._var_slices[k]]

            if data.shape[1] == 1:
                data = data[:, 0]

            data_dict[k] = xr.DataArray(data, self._var_coords[k], name=k)

        # unstack data in safe manner
        out = xr.Dataset(data_dict)
        for dim in ['features', 'samples']:
            try:
                out = out.unstack(dim)
            except ValueError:
                # unstack returns error if the dim is not an multiindex
                out = out.rename({dim: self.dims[dim][0]})

        return out


    def column_var(self, x):
        if x.ndim == 1:
            return x.data[:, None]
        else:
            return x


def test_datamatrix():
    from gnl.datasets import tiltwave

    # setup data
    a = tiltwave()
    b = a.copy()
    D = xr.Dataset({'a': a, 'b': b})

    mat = DataMatrix(['z'], ['x'], ['a', 'b'])
    y = mat.dataset_to_mat(D)

    x = mat.mat_to_dataset(y)

    for k in D.data_vars:
        np.testing.assert_allclose(D[k], x[k].transpose(*D[k].dims))
