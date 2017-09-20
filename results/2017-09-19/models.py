import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import  make_pipeline
from sklearn.linear_model import Ridge
from xnoah.data_matrix import  stack_cat
from mca import MCA

class NullFeatureRemover(TransformerMixin, BaseEstimator):

    def fit(self, x, y=None):
        x = np.asarray(x)
        std = x.std(axis=0)
        self.mask_ = std < 1e-5

        return self

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


def  _prepare_variable(x, do_scale, do_weight, scale, weight, sample_dims):

    if do_scale:
        x /= scale

    if do_weight:
        x *= np.sqrt(weight)

    return get_mat(x, sample_dims)

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
