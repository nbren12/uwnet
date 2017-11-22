#!/usr/bin/env python
"""Fit a torch model for

.. math::

    (x^n+1 - x^n)/dt - g =  f(x^n)

"""
import click
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from lib.models.torch_models import (Scaler, TorchRegressor,
                                     single_layer_perceptron, train, ConcatDataset, numpy_to_variable, predict)


def main(input, output, n):
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

    def loss_function(x, y):
        pred = net(x)
        return torch.mean(
            torch.pow(pred - y.float(), 2).mul(scale_weight.float()))

    # cast to double for numerical stability purposes
    x = X[:-1].reshape((-1, 68)).astype(float)
    xp = X[1:].reshape((-1, 68)).astype(float)
    g = G[:-1].reshape((-1, 68)).astype(float)

    sig = numpy_to_variable(x.std(axis=0))
    mu = numpy_to_variable(x.mean(axis=0))

    y = (xp - x) / dt - g

    # make a data loader
    dataset = ConcatDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=100, shuffle=True)

    net = nn.Sequential(Scaler(mu, sig), single_layer_perceptron(68, 68))
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

    torch.save(net, output)

    return net


if __name__ == '__main__':
    try:
        snakemake
    except NameError:
        main()
    else:
        main(snakemake.input[0], snakemake.output[0],
             snakemake.params.n)
