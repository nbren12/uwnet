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


def train(data_loader, net, loss_fn, optimizer=None, num_epochs=1):
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

            # use first variable for prediction
            x = data_args[0]
            pred = net(x)

            # pass all data args to loss_function
            loss = loss_fn(pred, *data_args)
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


class SupervisedProblem(Dataset):
    """"""

    def __init__(self, x, y):
        super(SupervisedProblem, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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
        def loss_function(pred, x, y):
            return self.loss_fn(pred.float(), y.float())

        train(
            data_loader,
            net,
            loss_function,
            optimizer=optim,
            num_epochs=self.num_epochs)

        self.net_ = net
        return self

    def predict(self, x):
        preds_np = self.net_(Variable(torch.FloatTensor(x))).data.numpy()
        return preds_np
