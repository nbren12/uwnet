from functools import partial

import attr
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from torch.autograd import Variable

from tqdm import tqdm


def batch_generator(x_train, y_train, batch_size=10):
    """Produce batches of x_train y_train

    This is useful for neural network training
    """
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    n = x_train.shape[0]
    inds = np.arange(x_train.shape[0])
    np.random.shuffle(inds)
    i = 0
    while (i + 1) * batch_size < n:
        b = i * batch_size
        e = min(b + batch_size, n)
        yield torch.FloatTensor(x_train[b:e]), torch.FloatTensor(y_train[b:e])
        i += 1


def _train(net, loss_fn, x_train, y_train, optimizer=None, num_epochs=2,
           shuffle=True):

    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=.0001)

    for epoch in range(num_epochs):
        avg_loss = 0
        if shuffle:
            inds = np.random.choice(x_train.shape[0],
                                    x_train.shape[0],
                                    replace=False)
        else:
            inds = np.arange(x_train.shape[0])

        generator = batch_generator(x_train[inds], y_train[inds],
                                    batch_size=100)
        for batch_idx, (x, y) in tqdm(enumerate(generator), total=10000):
            x, y = Variable(x), Variable(y)
            optimizer.zero_grad()  # this is not done automatically
            pred = net(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            avg_loss += loss.data.numpy()

        print(f"Epoch: {epoch} [{batch_idx}]\tLoss: {avg_loss}")
        avg_loss = 0


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

@attr.s
class TorchRegressor(BaseEstimator, RegressorMixin):

    net_fn = attr.ib()
    loss_fn = attr.ib()
    optim_cls = attr.ib(default=torch.optim.Adam)
    optim_kwargs = attr.ib(default=attr.Factory(dict))
    num_epochs = attr.ib(default=3)

    def fit(self, x, y):
        net = self.net_fn(x.shape[1], y.shape[1])
        optim = self.optim_cls(net.parameters(), **self.optim_kwargs)
        _train(
            net,
            self.loss_fn,
            x, y,
            optimizer=optim,
            num_epochs=self.num_epochs,
            shuffle=True)

        self.net_ = net
        return self

    def predict(self, x):
        preds_np = self.net_(Variable(torch.FloatTensor(x))).data.numpy()
        return preds_np


class single_layer_perceptron(nn.Module):
    def __init__(self, n_in, n_out, num_hidden=256):
        super(single_layer_perceptron, self).__init__()


        self.layers = nn.Sequential(
            nn.Linear(n_in, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, n_out)
        )

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
        return self.layers(x) #+ self.lin(x)
