import pandas as pd
import numpy as np
import xarray as xr

from sklearn.base import BaseEstimator, TransformerMixin

class Stacker(BaseEstimator, TransformerMixin):
    def __init__(self, feature_dims=()):
        self.feature_dims = feature_dims

    def fit(self, X, y=None):
        return self

    def transform(self, X: xr.DataArray):
        if not set(self.feature_dims) <= set(X.dims):
            raise ValueError(
                f"dims {self.feature_dims} is not a subset of input"
                "dimensions")

        dim_dict = {'samples': [dim for dim in X.dims
                                if dim not in self.feature_dims],
                    'features': self.feature_dims}

        data = X

        for new_dim, dims in dim_dict.items():
            if dims:
                data = data.stack(**{new_dim: dims})

            else:
                data = data.assign_coords(**{new_dim: 1}).expand_dims(new_dim)

        # store stacked coords for later use by inverse_transform
        # TODO this should be moved to fit
        self.coords_ = data.coords
        return data.transpose("samples", "features")

    def inverse_transform(self, X):
        """Inverse transform

        Assume X.shape = (m,nfeats)
        """
        m, n = X.shape

        samp_idx= pd.Index(np.arange(m), name='samples')
        coords = (samp_idx, self.coords_['features'])
        xarr = xr.DataArray(X, coords=coords)

        if len(self.feature_dims) == 0:
            raise NotImplementedError
        elif len(self.feature_dims) == 1:
            return xarr.rename({'features': self.feature_dims[0]})
        else:
            return xarr.unstack("features")



class Weighter(TransformerMixin):
    def __init__(self, w):
        self.weight = w

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.weight

    def inverse_transform(self, X):
        return X / self.weight


class WeightedNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, w=None):
        self.w = w

    def fit(self, X, y=None):
        w = self.w

        dims = [dim for dim in X.dims
                if dim not in w.dims]

        sig = X.std(dims)
        avg_var = (sig**2*w/w.sum()).sum(w.dims)
        self.x_scale_ = np.sqrt(avg_var)

        return self

    def transform(self, X):
        return X / self.x_scale_

    def inverse_transform(self, X):
        return X * self.x_scale_


class Select(BaseEstimator, TransformerMixin):
    def __init__(self, key=None, sel=None):
        self.key = key
        self.sel = {} if sel is None else sel

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X[self.key]
        if self.sel:
            out = out.sel(**self.sel)
        return out

    def inverse_transform(self, X):
        return X.to_dataset(name=self.key)

class XarrayMapper(BaseEstimator, TransformerMixin):
    def __init__(self,
                 features,
                 default=False,
                 sparse=False,
                 df_out=False,
                 input_df=False):
        """
        Params:
        features    a list of tuples with features definitions.
                    The first element is the pandas column selector. This can
                    be a string (for one column) or a list of strings.
                    The second element is an object that supports
                    sklearn's transform interface, or a list of such objects.
                    The third element is optional and, if present, must be
                    a dictionary with the options to apply to the
                    transformation. Example: {'alias': 'day_of_week'}
        """
        self.features = features

    def fit(self, X, y=None):
        for key, mod in self.features:
            mod.fit(X[key], y)

        return self

    def transform(self, X):

        out = []
        for key, mod in self.features:
            out.append(mod.transform(X[key]))
        return np.hstack(out)
