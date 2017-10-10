import numpy as np
import xarray as xr

from sklearn.base import BaseEstimator, TransformerMixin


class Stacker(BaseEstimator, TransformerMixin):
    def __init__(self, sample_dims):
        self.sample_dims = sample_dims

    def fit(self, X, y=None):
        return self

    def transform(self, X: xr.DataArray):
        feature_dims = [dim for dim in X.dims if dim not in self.sample_dims]

        if not set(self.sample_dims) <= set(X.dims):
            raise ValueError(
                f"Sample_dims {self.sample_dims} is not a subset of input"
                "dimensions")

        data = X.stack(samples=self.sample_dims)
        if feature_dims:
            data = data.stack(features=feature_dims)\
                       .transpose("samples", "features")\
                       .data

            return data
        else:
            return data.data[:, None]

    def inverse_transform(self, X):
        raise NotImplementedError


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
