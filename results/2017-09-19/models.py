import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import  make_pipeline
from sklearn.linear_model import Ridge
from xnoah.data_matrix import  stack_cat, unstack_cat
from mca import MCA


class NullFeatureRemover(TransformerMixin, BaseEstimator):

    def fit(self, x, y=None):
        x = np.asarray(x)
        std = x.std(axis=0)
        self.mask_ = std < 1e-10

        return self

    def transform(self, x, y=None):
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


class XarrayPreparer(TransformerMixin):
    def __init__(self, sample_dims=[], feauture_dims=[], weight=1.0, weight_input=False,
                 weight_output=True, scale_input=False, scale_output=False):

        self.sample_dims = sample_dims
        self.feature_dims = feauture_dims
        self.weight_input=weight_input
        self.weight_output=weight_output
        self.scale_input = scale_input
        self.scale_output = scale_output
        super(XarrayPreparer, self).__init__()


    def fit_transform(self, X, y=None):

        # normalize inputs
        self.scales_x_ = compute_weighted_scale(self.weight, self.sample_dims, X)

        if self.scale_input:
            X = self.scales_x_ * X

        if self.weight_input:
            X = X * np.sqrt(self.weight)


        # Xmat = get_mat(X, self.sample_dims, self.)


        return get_mat(in_ds), get_mat(out_ds)


RidgeRemover = make_pipeline(NullFeatureRemover(), Ridge(2, normalize=True))