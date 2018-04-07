"""This module contains routines for handling the intermediate representation
of the data. The routines in this module all have a dict `data` as the first
argument. This dict should have the same structure as produced by `prepare_data`.


See Also
--------
lib.data.prepare_data

"""
import numpy as np
import torch
from toolz import merge_with, valmap
from torch.autograd import Variable

from .datasets import DictDataset, WindowedData
from .loss import dynamic_loss
from .normalization import scaler


def _numpy_to_variable(x):
    return Variable(torch.FloatTensor(x))


def to_constants(data):
    return {
        'w': Variable(torch.FloatTensor(data['w']['sl'])),
        'z': Variable(torch.FloatTensor(data['z']))
    }


def to_scaler(data, cuda=False):
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


def _data_to_loss_feature_weights(data, cuda=True):
    def _f(args):
        w, scale = args
        return w / scale**2

    w = merge_with(_f, data['w'], data['scales'])
    w = valmap(_numpy_to_variable, w)

    if cuda:
        w = valmap(lambda x: x.cuda(), w)

    return w


def to_dynamic_loss(data, cuda=True, **kwargs):
    weights = _data_to_loss_feature_weights(data, cuda=cuda)
    return dynamic_loss(weights=weights, **kwargs)



def to_dataset(data,
               window_size,
               prognostic_variables=('sl', 'qt'),
               forcing_variables=('sl', 'qt', 'QRAD', 'LHF', 'SHF', 'Prec',
                                  'W', 'SOLIN')):


    X, G = data['X'], data['G']
    X = DictDataset({
        key: WindowedData(X[key], window_size)
        for key in prognostic_variables
    })
    G = DictDataset(
        {key: WindowedData(G[key], window_size)
         for key in forcing_variables})

    return DictDataset({'prognostic': X, 'forcing': G})


def get_num_features(data):
    return len(data['X']['p']) * 2
