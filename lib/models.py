from functools import partial
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.base import TransformerMixin, BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from xnoah.data_matrix import stack_cat, unstack_cat
from .mca import MCA




def weights_to_np(w, feats):
    idx = feats.indexes['features']

    def f(i):
        if i < 0:
            return 1.0
        else:
            return float(w.sel(z=idx.levels[1][i]))

    return xr.DataArray(
        np.array([f(i) for i in idx.labels[1]]), coords=(feats, ))


class WeightedOutput(object):
    """Light weight class for weighting the output of an sklearn function"""

    def __init__(self, model=None, weight=None, nfit=10000):
        self._mod = model
        self.w = np.asarray(weight)
        self.nfit = nfit

    def fit(self, x, y):
        n = x.shape[0]
        if self.nfit is None:
            rand = slice(None)
        else:
            rand = np.random.choice(n, self.nfit)
        self._mod.fit(x[rand], y[rand] * np.sqrt(self.w))
        return self

    def predict(self, x):
        return self._mod.predict(x) / np.sqrt(self.w)

    def score(self, x, y):
        return self._mod.score(x, y * np.sqrt(self.w))

    def __repr__(self):
        return "WeightedOutput(%s)" % repr(self._mod)

    def __getattr__(self, key):
        return getattr(self._mod, key)

    @classmethod
    def quickfit(cls, in_file, out_file, w_file, model=None, **kwargs):
        X = xr.open_dataset(in_file)
        Y = xr.open_dataset(out_file)
        w = xr.open_dataarray(w_file)
        xmat = prepvar(X, **kwargs)
        ymat = prepvar(Y, **kwargs)
        wmat = weights_to_np(w, ymat.features).data
        n, m = xmat.shape

        if model is None:
            model = LinearRegression()

        mod = WeightedOutput(model, wmat)
        mod.fit(xmat, ymat)

        return dict(x=xmat, y=ymat, w=wmat, mod=mod.fit(xmat, ymat))

def compute_dp(p):
    p_ghosted = np.hstack((2 * p[0] - p[1], p, 0))
    p_interface = (p_ghosted[1:] + p_ghosted[:-1]) / 2
    dp = -np.diff(p_interface)

    return dp


def plot_lrf(lrf,
             p,
             input_vars,
             output_vars,
             width_ratios=[1, 1, .3, .3],
             figsize=(10, 5),
             image_kwargs={}):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    p = np.asarray(p)
    ni, no = len(input_vars), len(output_vars)
    print("Making figure with", ni, "by", no, "panes")

    fig, axs = plt.subplots(no, ni, figsize=figsize)

    grid = gridspec.GridSpec(2, len(input_vars), width_ratios=width_ratios)

    for j, input_var in enumerate(input_vars):
        for i, output_var in enumerate(output_vars):
            ax = plt.subplot(grid[i, j])
            lrf_pane = np.asarray(lrf.loc[input_var][output_var])
            #             print(lrf_pane.shape)
            if lrf_pane.shape[0] == 1:
                ax.plot(lrf_pane.ravel(), p)
                ax.invert_yaxis()
            else:
                # compute pressure difference
                dp = compute_dp(p)
                im = ax.pcolormesh(p, p, (lrf_pane).T / dp, **image_kwargs)
                # make colorbar
                cbar = plt.colorbar(im, orientation='vertical', pad=.02)
                # invert y axis for pressure coordinates
                ax.invert_yaxis()
                ax.invert_xaxis()

            # turn off axes ylabels
            if j > 0:
                ax.set_yticks([])
            if i < no - 1:
                ax.set_xticks([])

            # Add variable names
            if j == 0:
                ax.set_ylabel(output_var)

            if i == 0:
                ax.set_title(input_var)

    return fig, axs


def get_lrf(mod, xmat, ymat):
    m = len(xmat.indexes['features'])
    lrf = mod.predict(np.eye(m)) - mod.predict(0 * np.eye(m))
    row, col = xmat.indexes['features'], ymat.indexes['features']
    return pd.DataFrame(lrf, index=row, columns=col)


def get_mat(ds, sample_dims=['x', 'time']):
    feature_dims = [dim for dim in ds.dims if dim not in sample_dims]

    return stack_cat(ds, 'features', feature_dims) \
        .stack(samples=sample_dims) \
        .transpose('samples', 'features')


def compute_weighted_scale(weight, sample_dims, ds):
    def f(data):
        sig = data.std(sample_dims)
        if set(weight.dims) <= set(sig.dims):
            sig = (sig**2 * weight/weight.sum()).sum(weight.dims).pipe(np.sqrt)
        return sig

    return ds.apply(f)


def mul_if_dims_subset(weight, x):
    if set(weight.dims) <= set(x.dims):
        return x * weight
    else:
        return x


def _prepare_variable(x, sample_dims, scale=None, weight=None):

    if scale is not None:
        x /= scale

    if weight is not None:
        weighter = partial(mul_if_dims_subset, np.sqrt(weight))
        x = x.apply(weighter)

    return get_mat(x, sample_dims)


def _unprepare_variable(y, sample_dims, scale=None, weight=None):

    y = unstack_cat(y, 'features').unstack('samples')

    if scale is not None:
        y *= scale

    if weight is not None:
        weighter = partial(mul_if_dims_subset, 1 / np.sqrt(weight))
        y = y.apply(weighter)

    return y


def _unstack(y, coords):
    return unstack_cat(xr.DataArray(y, coords), 'features') \
        .unstack('samples')


def _score_dataset(y_true, y, sample_dims, return_scalar=True):

    if not set(y.dims) >= set(sample_dims):
        raise ValueError("Sample dims must be a subset of data dimensions")

    # means
    ss = ((y_true - y_true.mean(sample_dims))**2).sum(sample_dims).sum()

    # prediction
    sse = ((y_true - y)**2).sum(sample_dims).sum()

    r2 = 1 - sse / ss
    sse_ = sse.to_array().sum()
    ss_ = ss.to_array().sum()

    scores = {'total': float(1.0 - sse_ / ss_)}
    for key in r2:
        scores[key] = float(r2[key])

    return scores


class XWrapper(object):
    def __init__(self, model, sample_dims, weight):
        self._model = model
        self.sample_dims = sample_dims
        self.weight = weight
        self.scales_x_ = None

    def fit(self, x, y):
        self.scales_x_ = compute_weighted_scale(self.weight, self.sample_dims,
                                                x)
        x, y = self.prepvars(x, y)
        self.yfeats = y.features
        self.xfeats = x.features
        self._model.fit(x, y)
        return self

    def predict(self, x):
        x = self.prepvars(x)
        y = self._model.predict(x)

        coords = (x.samples, self.yfeats)

        return _unprepare_variable(
            xr.DataArray(y, coords), self.sample_dims, weight=self.weight)

    def score(self, x, y):
        pred = self.predict(x)
        # need to weight true output
        weighter = partial(mul_if_dims_subset, np.sqrt(self.weight))
        y = y.apply(weighter)
        pred = pred.apply(weighter)

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
MyRidge = make_pipeline(Ridge(100.0, normalize=True))
MyRidge.prep_kwargs = dict(
    scale_input=True,
    scale_output=False,
    weight_input=True,
    weight_output=True)
MyRidge.param_grid = {'ridge__alpha': np.logspace(-10, 3, 15)}
# MyRidge.param_grid = {'ridge__alpha': [.19]}

# This dictionary is used by the Snakefile
model_dict = {'ridge': MyRidge}
