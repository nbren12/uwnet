import torch
from toolz import curry
from torch.autograd import Variable


def _numpy_to_variable(x):
    return Variable(torch.FloatTensor(x))


def _scale_var(scale, mean, x):
    x = x.double()
    mu = mean.double()
    sig = scale.double()

    x = x.sub(mu)
    x = x.div(sig + 1e-7)

    return x.float()


@curry
def scaler(scales, means, x):
    out = {}
    for key in x:
        if key in scales and key in means:
            out[key] = _scale_var(scales[key], means[key], x[key])
        else:
            out[key] = x[key]
    return out
