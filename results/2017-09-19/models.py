from functools import partial
import numpy as np
import xarray as xr
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from xnoah.data_matrix import stack_cat, unstack_cat
from mca import MCA

class NullFeatureRemover(TransformerMixin, BaseEstimator):

    def fit(self, x, y=None):
        x = np.asarray(x)
        std = x.std(axis=0)
        self.mask_ = std < 1e-3

        return self

    @property
    def nnz(self):
        return (-self.mask_).sum()

    def transform(self, x, y=None):
        if self.mask_.shape[0] != x.shape[1]:
            raise ValueError("X is not the same shape as the stored mask")
        x = np.asarray(x)
        xt = x[:,-self.mask_]

        if y is None:
            return xt
        else:
            return xt, y


def get_mat(ds, sample_dims=['x', 'time']):
    feature_dims = [dim for dim in ds.dims
                    if dim not in sample_dims]

    return stack_cat(ds, 'features', feature_dims) \
        .stack(samples=sample_dims) \
        .transpose('samples', 'features')


def compute_weighted_scale(weight, sample_dims, ds):
    def f(data):
        sig = data.std(sample_dims)
        if set(weight.dims) <= set(sig.dims):
            sig = (sig ** 2 * weight).sum(weight.dims).pipe(np.sqrt)
        return sig
    return ds.apply(f)


def mul_if_dims_subset(weight, x):
    if set(weight.dims) <= set(x.dims):
        return x * np.sqrt(weight)
    else:
        return x

def  _prepare_variable(x, sample_dims, scale=None, weight=None):

    if scale is not None:
        x /= scale

    weighter = partial(mul_if_dims_subset, weight)

    if weight is not None:
        x = x.apply(weighter)

    return get_mat(x, sample_dims)

def _unstack(y, coords):
    return unstack_cat(xr.DataArray(y, coords), 'features') \
        .unstack('samples')


def _score_dataset(y_true, y, sample_dims, return_scalar=True):

    if not set(y.dims)  >= set(sample_dims):
        raise ValueError("Sample dims must be a subset of data dimensions")

    # means
    ss = ((y_true - y_true.mean(sample_dims))**2).sum(sample_dims).sum()

    # prediction
    sse = ((y_true - y)**2).sum(sample_dims).sum()

    r2 = 1- sse/ss
    sse_ = sse.to_array().sum()
    ss_ = ss.to_array().sum()

    return float(1- sse_/ss_), float(r2.q1), float(r2.q2)


class XWrapper(object):
    def __init__(self, model, sample_dims, weight):
        self._model = model
        self.sample_dims = sample_dims
        self.weight = weight
        self.scales_x_ = None

    def fit(self, x, y):
        self.scales_x_ = compute_weighted_scale(self.weight,
                                                self.sample_dims,
                                                x)
        x, y = self.prepvars(x, y)
        self.yfeats = y.features
        self._model.fit(x,y)
        return self

    def predict(self, x):
        x = self.prepvars(x)
        y =  self._model.predict(x)

        coords = (x.samples, self.yfeats)
        return unstack_cat(xr.DataArray(y, coords), 'features') \
            .unstack('samples')

    def score(self, x, y):
        pred = self.predict(x)
        # need to weight true output
        weighter = partial(mul_if_dims_subset, self.weight)
        y = y.apply(weighter)

        return _score_dataset(y, pred, self.sample_dims)

    def set_params(self, **kwargs):
        return self._model.set_params(**kwargs)

    def prepvars(self, X, y=None):
        weight = self.weight
        scales_x_ = self.scales_x_
        sample_dims = self.sample_dims
        X = _prepare_variable(X, sample_dims, scales_x_, weight)
        if y is not None:
            y = _prepare_variable(y, sample_dims, weight=weight)

            return X, y
        else:
            return X


# Ridge Regression
MyRidge = make_pipeline(Ridge(1.0, normalize=True))
MyRidge.prep_kwargs = dict(scale_input=False, scale_output=False,
                           weight_input=True, weight_output=True)
MyRidge.param_grid = {'ridge__alpha': [.19]}

# MCA
_MCA = make_pipeline(MCA(), LinearRegression())
_MCA.prep_kwargs = dict(scale_input=True, scale_output=False,
                           weight_input=True, weight_output=True)
_MCA.param_grid = {'mca__n_components': [1, 2, 4, 5, 6, 7, 8, 9,
                                         10, 20, 30, 40, 50, 100, 120]}

# Principal component regression
# PCA automatically demeans the data, which is what we want to happen
_PCR = make_pipeline(PCA(), LinearRegression())
_PCR.prep_kwargs = dict(scale_input=True, scale_output=False,
                        weight_input=True, weight_output=True)
_PCR.param_grid = {'pca__n_components': [1, 2, 4, 5, 6, 7, 8, 9,
                                         10, 20, 30, 40, 50, 100, 120]}
# This dictionary is used by the Snakefile
model_dict = {
    'ridge': MyRidge,
    'mca': _MCA,
    'pcr': _PCR
}
