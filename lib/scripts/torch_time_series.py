#!/usr/bin/env python
"""Fit a torch model for

.. math::

    (x^n+1 - x^n)/dt - g =  f(x^n)
    x^n+1 = x^n + dt * (f(x^n) + g)

"""
import click
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from lib.models.torch_models import (
    Scaler, TorchRegressor, single_layer_perceptron, train,
    numpy_to_variable, predict, ConcatDataset,
    EulerStepper)



@click.command()
@click.argument("input")
@click.argument("output")
@click.option("-n", default=1)
@click.option("--learning-rate", default=.001)
@click.option("--nsteps", default=1)
@click.option("--nhidden", default=256)
def main(*args, **kwargs):
    train_euler_network(*args, **kwargs)

def train_euler_network(input, output, n, nsteps=1,
                        learning_rate=.001, nhidden=256):
    num_epochs = n
    data = np.load(input)

    X = data['X']
    G = data['G']
    scale = data['scales']
    w = data['w']

    # the sampling interval of the data
    dt = 3 / 24

    # define the loss function
    scale_weight = numpy_to_variable(w / scale**2)

    def forward(x, xp, g):
        return stepper(x, g)

    def loss_function(x, xp, g):
        pred = forward(x, xp, g)
        return torch.mean(
            torch.pow(pred - xp.float(), 2).mul(scale_weight.float()))

    # cast to double for numerical stability purposes
    x = X[:-1].reshape((-1, 68)).astype(float)
    xp = X[1:].reshape((-1, 68)).astype(float)
    g = G[:-1].reshape((-1, 68)).astype(float)

    sig = numpy_to_variable(x.std(axis=0))
    mu = numpy_to_variable(x.mean(axis=0))

    # make a data loader
    ntrain = 100000
    inds = np.random.choice(x.shape[0], ntrain)
    x, xp, g= [x[inds] for x in [x, xp, g]]
    dataset = ConcatDataset(x, xp, g)
    data_loader = DataLoader(dataset, batch_size=100, shuffle=True)

    net = nn.Sequential(
        Scaler(mu, sig),
        single_layer_perceptron(68, 68, num_hidden=nhidden)
    )
    stepper = EulerStepper(net, nsteps, dt)
    optimizer = torch.optim.Adam(net.parameters())


    # train the model
    train(
        data_loader,
        loss_function,
        optimizer=optimizer,
        num_epochs=num_epochs)

    # plot output for one location
    x = X[:, 8, 0, :]
    pred = predict(net, x)

    torch.save(stepper, output)

    return net


if __name__ == '__main__':
    main()
