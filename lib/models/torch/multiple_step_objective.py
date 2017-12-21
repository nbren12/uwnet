"""Fit model for the multiple time step objective function. This requires some special dataloaders etc

"""
from functools import partial
import attr
import click
import toolz
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from .torch_datasets import ConcatDataset, WindowedData
from .torch_models import (numpy_to_variable, single_layer_perceptron, Scaler,
                           EulerStepper, train)

def _numpy_to_variable(x):
    return Variable(torch.FloatTensor(x))


def _data_to_torch_dataset(data, window_size):
    X, G = data['X'], data['G']
    dataset = ConcatDataset(WindowedData(X, chunk_size=window_size),
                            WindowedData(G, chunk_size=window_size))

    return dataset


def _data_to_scaler(data):
    # compute mean and stddev
    # this is an error, std does not work like this
    X = data['X']
    m = X.shape[-1]
    mu = X.reshape((-1, m)).mean(axis=0)
    sig = X.reshape((-1, m)).std(axis=0)
    mu, sig = [_numpy_to_variable(np.squeeze(x)) for x in [mu, sig]]
    return Scaler(mu, sig)


def _data_to_loss_feature_weights(data):
    return _numpy_to_variable(data['w']/data['scales']**2)


def weighted_loss(x, y, scale_weight):
    return torch.mean(torch.pow(x-y, 2).mul(scale_weight.float()))


@curry
def multiple_step_mse(stepper, feature_weight, time_weight, x, g):
    """Weighted MSE loss accumulated over multiple time steps

    Parameters
    ----------
    stepper : nn.Module
        torch stepper
    feature_weight :
        weights for loss function
    time_weight : torch.Tensor
        time_weight[i] gives the weight at step i
    x : (batch, time, feat)
        torch tensor of prognostic variables
    g : (batch, time, feat)
        torch tensor of forcings
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
        loss += weighted_loss(xiter, xactual, feature_weight) * time_weight[i-1]

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
    scaler = _data_to_scaler(data)
    feature_weight = _data_to_loss_feature_weights(data)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # compute weights for loss function
    # exponential decay in loss
    # loss_weight = torch.exp(-torch.arange(0, window_size-1) * dt.data/2)
    # equally weighted
    time_weight = torch.ones(window_size-1)
    time_weight /= torch.sum(time_weight)
    time_weight = Variable(time_weight)

    # define the neural network
    m = feature_weight.size(0)
    net = nn.Sequential(
        scaler,
        single_layer_perceptron(m, m, num_hidden=nhidden),
        )
    stepper = EulerStepper(net, nsteps=nsteps, h=dt)

    # _init_linear_weights(net, .01/nsteps)
    loss_function = multiple_step_mse(stepper,
                                      _data_to_loss_feature_weights(data),
                                      time_weight)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # train the model
    train(data_loader, loss_function, optimizer=optimizer,
          num_epochs=num_epochs, num_steps=num_steps)

    return stepper
