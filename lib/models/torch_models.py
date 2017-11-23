"""Module with all the torch code

Routine Listings
-------
train  : function for training
Scaler : preprocessing torch module
SupervisedProblem : torch dataset class for supervised learning problems
TorchRegressor : Sklearn Torch wrapper
"""
from functools import partial

import attr
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from torch.autograd import Variable

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def train(data_loader, loss_fn, optimizer=None, num_epochs=1):
    """Train a torch model"""

    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(num_epochs):
        avg_loss = 0

        for batch_idx, data in tqdm(
                enumerate(data_loader), total=len(data_loader)):
            optimizer.zero_grad()  # this is not done automatically in torch

            # cast all variables to Variable
            data_args = [Variable(x) for x in data]

            # pass all data args to loss_function
            loss = loss_fn(*data_args)

            loss.backward()
            optimizer.step()

            avg_loss += loss.data.numpy()

        avg_loss /= len(data_loader)
        print(f"Epoch: {epoch} [{batch_idx}]\tLoss: {avg_loss}")


class Scaler(nn.Module):
    """Scaled single layer preceptron

    Examples
    --------

    >>> a = torch.rand(1000,10)
    >>> scaler = Scaler(a.mean(0), a.std(0))
    >>> b = scaler(a)
    >>> b.mean(0)
    >>> b.std(0)

    1.0000
    1.0000
    1.0000
    1.0000
    1.0000
    1.0000
    1.0000
    1.0000
    1.0000
    1.0000
    [torch.FloatTensor of size 10]

    """

    def __init__(self, mu, sig):
        "docstring"
        super(Scaler, self).__init__()
        self.mu = mu
        self.sig = sig

    def forward(self, x):
        return x.sub(self.mu).div(self.sig + 1e-7).float()


class ResidualBlock(nn.Module):
    """Simple residual net block"""

    def __init__(self, n):
        super(ResidualBlock, self).__init__()
        self.lin1 = nn.Linear(n, n)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(n, n)

    def forward(self, x):
        y = self.lin2(self.relu(self.lin1(x)))

        return self.relu(y.add(x))


class single_layer_perceptron(nn.Module):
    def __init__(self, n_in, n_out, num_hidden=256):
        super(single_layer_perceptron, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_in, num_hidden),
            nn.ReLU(), nn.Linear(num_hidden, n_out))

        self.lin = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.layers(x) + self.lin(x)


class residual_net(nn.Module):
    def __init__(self, xshape, yshape, num_hidden=20):
        super(residual_net, self).__init__()

        _, nx = xshape
        _, ny = yshape

        self.layers = nn.Sequential(
            nn.Linear(nx, num_hidden),
            ResidualBlock(num_hidden),
            ResidualBlock(num_hidden), nn.Linear(num_hidden, ny))

        # self.lin = nn.Linear(nx, ny)

    def forward(self, x):
        return self.layers(x)  #+ self.lin(x)


class EulerStepper(nn.Module):
    """Module for performing n forward euler time steps of a neural network
    """
    def __init__(self, net, nsteps, h):
        super(EulerStepper, self).__init__()
        self.net = net
        self.nsteps = nsteps
        self.h = h

    def forward(self, x):
        nsteps = self.nsteps
        h = self.h
        net = self.net

        reduction = 1e4

        for i in range(nsteps):
            x = x.float() + h/nsteps * (net(x.double())/reduction)

        return x


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


@attr.s
class TorchRegressor(BaseEstimator, RegressorMixin):
    """SKLearn Wrapper around torch modules"""

    net_fn = attr.ib()
    loss_fn = attr.ib()
    optim_cls = attr.ib(default=torch.optim.Adam)
    optim_kwargs = attr.ib(default=attr.Factory(dict))
    num_epochs = attr.ib(default=1)
    batch_size = attr.ib(default=50)

    def fit(self, x, y):
        net = self.net_fn(x.shape[1], y.shape[1])
        optim = self.optim_cls(net.parameters(), **self.optim_kwargs)

        data_loader = DataLoader(SupervisedProblem(x, y),
                                 batch_size=self.batch_size,
                                 shuffle=True)

        # needed to add an argument for passing to
        # train
        def loss_function(x, y):
            pred = net(x)
            return self.loss_fn(pred.float(), y.float())

        train(
            data_loader,
            loss_function,
            optimizer=optim,
            num_epochs=self.num_epochs)

        self.net_ = net
        return self

    def predict(self, x):
        preds_np = self.net_(Variable(torch.FloatTensor(x))).data.numpy()
        return preds_np


def numpy_to_variable(x):
    return Variable(torch.DoubleTensor(x))


def predict(net, *args):

    torch_args = [numpy_to_variable(x.astype(float))
                  for x in args]
    return net(*torch_args).data.numpy()


def train_euler_network(input, n=1, nsteps=1,
                        learning_rate=.001, nhidden=256):
    """Train a single layer perceptron euler time stepping model
    
    For one time step this torch models performs the following math

    .. math::

        x^n+1 = x^n + dt * (f(x^n) + g)

    
    """
    num_epochs = n
    data = np.load(input)

    X = data['X']
    G = data['G']
    scale = data['scales']
    w = data['w']

    # the sampling interval of the data
    dt = Variable(torch.FloatTensor([3 / 24]))

    # define the loss function
    scale_weight = numpy_to_variable(w / scale**2)

    def forward(x, xp, g):
        return stepper(x)

    def loss_function(x, xp, g):
        pred = forward(x, xp, g)
        g = g.float()
        return torch.mean(
            torch.pow(pred + g.mul(dt) - xp.float(), 2).mul(scale_weight.float()))

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
    
    return stepper
