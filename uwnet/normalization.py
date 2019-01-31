import logging
from functools import reduce

from toolz.curried import valmap

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _numpy_to_variable(x):
    return torch.tensor(x).float()


def _scale_var(scale, mean, x):
    x = x.double()
    mu = mean.double()
    sig = scale.double()

    x = x.sub(mu)
    x = x.div(sig + 1e-7)

    return x.float()


def scaler(scales, means, x):
    out = {}
    for key in x:
        if key in scales and key in means:
            out[key] = _scale_var(scales[key], means[key], x[key])
        else:
            out[key] = x[key]
    return out


def _dict_to_parameter_dict(x):
    out = {}
    for key in x:
        out[key] = nn.Parameter(x[key], requires_grad=False)
    return nn.ParameterDict(out)


def _get_scaler_args_numpy(dataset):
    logger.info("Computing mean")
    mean = dataset.mean(['x', 'y', 'time'])
    logger.info("Computing std")
    scale = dataset.std(['x', 'y', 'time'])
    return mean, scale


def get_mean_scale(dataset):
    from .datasets import _ds_slice_to_torch
    out = map(_ds_slice_to_torch, _get_scaler_args_numpy(dataset))
    return map(valmap(torch.squeeze), out)


def add_tuple(x, y):
    return tuple(xx + yy for xx, yy in zip(x, y))


def moments(batch):
    data = batch.data.double()
    shape = data['QT'].shape
    num_examples = shape[1] * shape[0]
    m1 = data.sum(0).sum(0)
    m2 = (data**2).sum(0).sum(0)
    return num_examples, m1, m2


def moments_from_data_loader(loader):
    n, m1, m2 = reduce(add_tuple, map(moments, iter(loader)))
    m1 = m1 / n
    v = m2 / n - m1**2
    return m1, v.sqrt()


class Scaler(nn.Module):
    """Torch class for normalizing data along the final dimension"""

    def __init__(self, mean=None, scale=None):
        "docstring"
        super(Scaler, self).__init__()

    def forward(self, x):
        out = {}
        for key in x:
            if key in self.scale and key in self.mean:
                out[key] = _scale_var(self.scale[key], self.mean[key], x[key])
            else:
                out[key] = x[key]
        return out

    def set_mean_scale(self, mean, scale):
        self.mean = _dict_to_parameter_dict(mean)
        self.scale = _dict_to_parameter_dict(scale)

    def fit_xarray(self, dataset):
        mean, scale = get_mean_scale(dataset)
        self.set_mean_scale(mean, scale)
        return self

    def fit_generator(self, data_loader):
        mean, sig = moments_from_data_loader(data_loader)
        self.set_mean_scale(mean.view(-1), sig.view(-1))
        return self
