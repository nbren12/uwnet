"""Fit model for the multiple time step objective function. This requires some special dataloaders etc

"""
from functools import partial
import attr
import click
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from .torch_datasets import ConcatDataset, WindowedData
from .torch_models import (numpy_to_variable, single_layer_perceptron, Scaler,
                           EulerStepper, train)


def _init_linear_weights(net, std):
    for mod in net.modules():
        if isinstance(mod, nn.Linear):
            mod.weight.data.normal_(std=std)


def _stack_dict_arrs(d, keys, axis=-1):
    return np.concatenate([d[key] for key in keys], axis=-1)

def _data_to_torch_dataset(data, window_size):

    X = _stack_dict_arrs(data['X'], data['fields'])
    G = _stack_dict_arrs(data['G'], data['fields'])

    # do not use the moisture field above 200 hPA
    # this is the top 14 grid points for NGAqua
    ntop = -14
    X = X[..., :ntop].astype(float)
    G = G[..., :ntop].astype(float)

    dataset = ConcatDataset(WindowedData(X, chunk_size=window_size),
                            WindowedData(G, chunk_size=window_size))

    return dataset


def _data_to_stats(data):

    X = _stack_dict_arrs(data['X'], data['fields'])
    G = _stack_dict_arrs(data['G'], data['fields'])

    scale = data['scales']
    w = data['w']

    # do not use the moisture field above 200 hPA
    # this is the top 14 grid points for NGAqua
    ntop = -14
    X = X[..., :ntop].astype(float)
    G = G[..., :ntop].astype(float)
    scale = scale[:ntop].astype(float)
    w = w[:ntop].astype(float)

    # scaling and other code
    scale_weight = w / scale**2

    # compute mean and stddev
    # this is an error, std does not work like this
    m = X.shape[-1]
    mu = X.reshape((-1, m)).mean(axis=0)
    sig = X.reshape((-1, m)).std(axis=0)

    mu, sig, scale_weight = [Variable(torch.FloatTensor((np.squeeze(x))))
                             for x in [mu, sig, scale_weight]]

    return mu, sig, scale_weight


def weighted_loss(x, y, scale_weight):
    return torch.mean(torch.pow(x-y, 2).mul(scale_weight.float()))


def multiple_step_mse(stepper, scale_weight, x, g):
    """Weighte MSE loss accumulated over multiple time steps
    """
    x = Variable(x.float())
    g = Variable(g.float())

    batch_size, window_size, nf = x.size()

    loss = 0
    xiter = x[:, 0, :]

    dt = stepper.h

    for i in range(1, window_size):
        # use trapezoid rule for the forcing
        xiter = stepper(xiter) + (g[:, i-1, :] + g[:, i, :]).mul(dt/2)
        xactual = x[:, i, :]
        loss += weighted_loss(xiter, xactual, scale_weight)

    if np.isnan(loss.data.numpy()):
        raise FloatingPointError

    return loss


def train_multistep_objective(data, num_epochs=1, num_steps=None, nsteps=1, learning_rate=.001,
                              nhidden=10, weight_decay=.0, ntrain=None, batch_size=100,
                              window_size=2):
    """Train a single layer perceptron euler time stepping model

    For one time step this torch models performs the following math

    .. math::

        x^n+1 = x^n + dt * (f(x^n) + g)


    """

    # the sampling interval of the data
    dt = Variable(torch.FloatTensor([3 / 24]))

    dataset = _data_to_torch_dataset(data, window_size)
    mu, sig, scale_weight = _data_to_stats(data)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # define the neural network
    m = mu.size(0)
    net = nn.Sequential(
        Scaler(mu, sig),
        single_layer_perceptron(m, m, num_hidden=nhidden),
        )
    stepper = EulerStepper(net, nsteps=nsteps, h=dt)

    # _init_linear_weights(net, .01/nsteps)
    loss_function = partial(multiple_step_mse, stepper, scale_weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # train the model
    train(data_loader, loss_function, optimizer=optimizer,
          num_epochs=num_epochs, num_steps=num_steps)

    return stepper
