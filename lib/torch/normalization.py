import numpy as np
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


def data_to_scalar(data, cuda=False):
    # compute mean and stddev
    # this is an error, std does not work like this
    means = {}
    scales = {}
    for key in data['X']:
        X = data['X'][key]
        m = X.shape[-1]
        mu = X.reshape((-1, m)).mean(axis=0)
        sig = X.reshape((-1, m)).std(axis=0)

        # convert to torch
        mu, sig = [_numpy_to_variable(np.squeeze(x)) for x in [mu, sig]]
        sig = torch.mean(sig)

        if cuda:
            mu = mu.cuda()
            sig = sig.cuda()

        means[key] = mu
        scales[key] = sig

    return scaler(scales, means)
