import logging

import torch
from toolz import curry
from toolz.curried import valmap
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


class Scaler(nn.Module):
    """Torch class for normalizing data along the final dimension"""

    def __init__(self, mean=None, scale=None):
        "docstring"
        super(Scaler, self).__init__()
        if mean is None:
            mean = {}
        if scale is None:
            scale = {}
        self.mean = _dict_to_parameter_dict(mean)
        self.scale = _dict_to_parameter_dict(scale)

    def forward(self, x):
        out = {}
        for key in x:
            if key in self.scale and key in self.mean:
                out[key] = _scale_var(self.scale[key], self.mean[key], x[key])
            else:
                out[key] = x[key]
        return out


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
