"""Module with all the torch code

Routine Listings
-------
train  : function for training
Scaler : preprocessing torch module
SupervisedProblem : torch dataset class for supervised learning problems
TorchRegressor : Sklearn Torch wrapper
"""
from functools import partial
from itertools import islice

import attr
import numpy as np
from scipy.linalg import logm
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from torch.autograd import Variable

from torch.utils.data import DataLoader
from tqdm import tqdm

from .torch_datasets import ConcatDataset


def train(data_loader, loss_fn, optimizer, num_epochs=1,
          num_steps=None):
    """Train a torch model

    Parameters
    ----------
    num_epochs : int
        number of epochs of training
    num_steps : int or None
        maximum number of batches per epoch. If None, then the full dataset is used.

    """

    if num_steps == None:
        num_steps = len(data_loader)

    for epoch in range(num_epochs):
        avg_loss = 0

        data_generator = islice(data_loader, num_steps)

        counter = 0
        for batch_idx, data in tqdm(enumerate(data_generator),
                                    total=num_steps):
            optimizer.zero_grad()  # this is not done automatically in torch

            # pass all data args to loss_function
            loss = loss_fn(data)

            loss.backward()
            optimizer.step()

            avg_loss += loss.data.cpu().numpy()
            counter += 1

        avg_loss /= counter
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

    def __init__(self, mu, sig, scale=True):
        "docstring"
        super(Scaler, self).__init__()
        self.mu = mu
        self.sig = sig
        self.scale = scale

    def forward(self, x):
        # need to cast to double to avoid large floating point errors when
        # subtracting the mean

        x = x.double()
        mu = self.mu.double()
        sig = self.sig.double()

        x = x.sub(mu)
        if self.scale:
            x = x.div(sig + 1e-7)

        return x.float()


class Subset(nn.Module):
    """Remove certain indices from final dimension

    This is useful for removing null-variance features
    """
    def __init__(self, inds):
        self.inds = inds

    def forward(self, x):
        return x[..., self.inds]


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

    def __init__(self, rhs, nsteps, h):
        super(EulerStepper, self).__init__()
        self.rhs = rhs
        self.nsteps = nsteps
        self.h = h

    def forward(self, args):
        x = args[0]
        nsteps = self.nsteps
        h = self.h
        rhs = self.rhs

        for i in range(nsteps):
            x = x + h / nsteps * rhs(args)

        return x

    def linear_response_function(self, x):
        """Compute the instantanious linear response function

        .. math::

            M = logm(I + h/n * rhs(x)) / (h/n)
        """
        I = np.eye(x.shape[0])
        h = self.h/self.nsteps
        h = h.data.numpy()


        jac = jacobian(self.rhs, x)
        return logm(I + h * jac)/h


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

        data_loader = DataLoader(
            SupervisedProblem(x, y), batch_size=self.batch_size, shuffle=True)

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


def jacobian(net, x):
    """Compute the Jacobian of a torch module with respect to its inputs"""
    x0 = torch.FloatTensor(np.squeeze(x)).double()
    x0 = Variable(x0, requires_grad=True)

    nv = x0.size(0)

    jac = torch.zeros(nv, nv)

    for i in range(nv):
        outi = net(x0)[i]
        outi.backward()
        jac[i, :] = x0.grad.data
        x0.grad.data.zero_()

    return jac.numpy()
