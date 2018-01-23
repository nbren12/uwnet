"""Fit model for the multiple time step objective function. This requires some special dataloaders etc

"""
from toolz import curry
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


def _data_to_scaler(data, cuda=False):
    # compute mean and stddev
    # this is an error, std does not work like this
    X = data['X']
    m = X.shape[-1]
    mu = X.reshape((-1, m)).mean(axis=0)
    sig = X.reshape((-1, m)).std(axis=0)
    mu, sig = [_numpy_to_variable(np.squeeze(x)) for x in [mu, sig]]
    sig = torch.mean(sig)

    if cuda:
        mu = mu.cuda()
        sig = sig.cuda()

    return Scaler(mu, sig)


def _data_to_loss_feature_weights(data, cuda=True):
    w = _numpy_to_variable(data['w']/data['scales']**2)
    if cuda:
        w = w.cuda()

    return w

@curry
def weighted_loss(weight, x, y):
    return torch.mean(torch.pow(x-y, 2).mul(weight.float()))


class ForcedStepper(nn.Module):
    def __init__(self, step, dt):
        super(ForcedStepper, self).__init__()
        self.step = step
        self.dt = dt

    def forward(self, x, g):
        """

        Parameters
        ----------
        x : (seq_len, batch, input_size)
            tensor containing prognostic variables
        g : (seq_len, batch, input_size)
            tensor containing forcings
        """
        window_size = x.size(0)

        xiter = x[0]
        steps = [xiter]

        for i in range(1, window_size):
            # use trapezoid rule for the forcing
            xiter = self.step(xiter) + (g[i-1] + g[i]).mul(self.dt/2)
            steps.append(xiter)

        return torch.stack(steps)


def mlp(layer_sizes):
    layers = []
    n = len(layer_sizes)
    for k in range(n-1):
        layers.append(nn.Linear(layer_sizes[k], layer_sizes[k+1]))
        if k < n - 2:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class RHS(nn.Module):
    def __init__(self, m, hidden=()):
        super(RHS, self).__init__()
        self.mlp = mlp((m,) + hidden + (m,))
        self.lin = nn.Linear(m, m, bias=False)

    def forward(self, x):
        return self.mlp(x) + self.lin(x)

def train_multistep_objective(data, num_epochs=4, window_size=10,
                              num_steps=500, batch_size=100, lr=0.01,
                              weight_decay=0.0, nsteps=1, nhidden=(10, 10, 10),
                              cuda=False):
    """Train a single layer perceptron euler time stepping model

    For one time step this torch models performs the following math

    .. math::

        x^n+1 = x^n + dt * (f(x^n) + g)


    """
    torch.manual_seed(1)

    # the sampling interval of the data
    dt = Variable(torch.FloatTensor([3 / 24]), requires_grad=False)
    if cuda:
        dt = dt.cuda()

    dataset = _data_to_torch_dataset(data, window_size)
    scaler = _data_to_scaler(data, cuda=cuda)
    feature_weight = _data_to_loss_feature_weights(data, cuda=cuda)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # define the neural network
    m = feature_weight.size(0)

    modules = [scaler]

    rhs = RHS(m, hidden=nhidden)
    net = nn.Sequential(
        scaler,
        rhs
    )
    stepper = EulerStepper(net, nsteps=nsteps, h=dt)
    nstepper = ForcedStepper(stepper, dt)
    loss = weighted_loss(feature_weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    if cuda:
        nstepper.cuda()

    # _init_linear_weights(net, .01/nsteps)
    def closure(x, g):
        x = Variable(x.float())
        g = Variable(g.float())

        if cuda:
            x, g = x.cuda(), g.cuda()

        x = x.transpose(0, 1)
        g = g.transpose(0, 1)

        y = nstepper(x, g)
        return loss(y, x)

    # train the model
    train(data_loader, closure, optimizer=optimizer,
          num_epochs=num_epochs, num_steps=num_steps)

    return nstepper



