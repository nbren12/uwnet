import numpy as np
import xarray as xr
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from xnoah.data_matrix import stack_cat, unstack_cat
from .mca import MCA

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
        xt = x[:,~self.mask_]

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


def  _prepare_variable(x, do_scale, do_weight, scale, weight, sample_dims):

    if do_scale:
        x /= scale

    def mul_if_dims_subset(x):
        if set(weight.dims) <= set(x.dims):
            return x * np.sqrt(weight)
        else:
            return x

    if do_weight:
        x = x.apply(mul_if_dims_subset)

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

    scores = {'total': float(1.0- sse_/ss_)}
    for key in r2:
        scores[key] = float(r2[key])

    return scores


class XarrayPreparer(TransformerMixin):
    """Object for preparing data for input

    I am not sure if this class is necessary or not. I implemented it to plug into the sklearn pipeline infrastructure,
    but sklearn pipelines do not apply transformations to the output variables, so it isn't really very helpful.
    """
    def __init__(self, sample_dims=(), weight=1.0, weight_input=False,
                 weight_output=True, scale_input=False, scale_output=False):

        self.sample_dims = sample_dims
        self.weight = weight
        self.weight_input=weight_input
        self.weight_output=weight_output
        self.scale_input = scale_input
        self.scale_output = scale_output
        super(XarrayPreparer, self).__init__()

    def fit(self, X, y=None):
        self.scales_x_ = compute_weighted_scale(self.weight, self.sample_dims, X)
        if y is not None:
            self.scales_y_ = compute_weighted_scale(self.weight, self.sample_dims, y)
        return self

    def transform(self, X, y=None):
        # normalize inputs
        X = _prepare_variable(X, self.scale_input, self.weight_input,
                              self.scales_x_, self.weight, self.sample_dims)
        self.input_coords_ = X.coords

        if y is not None:
            y = _prepare_variable(y, self.scale_input, self.weight_input,
                                  self.scales_y_, self.weight, self.sample_dims)
            return X, y

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X,y)

    def score(xprep, y_pred, y):

        # turn into Dataset
        y_true_dataset = _unstack(y, y.coords)
        y_pred_dataset = _unstack(y_pred, y.coords)


        # This might be unnessary
        # I think this might just be removed in the R2 calculation
        # also weighting would be removed
        if xprep.scale_output:
            y_true_dataset *= xprep.scale_y_
            y_pred_dataset *= xprep.scale_y_


        return _score_dataset(y_true_dataset, y_pred_dataset, xprep.sample_dims)


# Ridge Regression
MyRidge = make_pipeline(Ridge(100.0, normalize=False))
MyRidge.prep_kwargs = dict(scale_input=True, scale_output=False,
                           weight_input=True, weight_output=True)
MyRidge.param_grid = {'ridge__alpha': np.logspace(-10, 3, 15)}
# MyRidge.param_grid = {'ridge__alpha': [.19]}

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
