"""Fit model for the multiple time step objective function. This requires some special dataloaders etc

"""
from toolz import curry, first, valmap, merge_with, assoc
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .torch_datasets import ConcatDataset, WindowedData
from .torch_models import train


# list of variables used
prognostic_variables = ['sl', 'qt']
forcing_variables = 'sl qt QRAD LHF SHF Prec'.split(' ')


def _numpy_to_variable(x):
    return Variable(torch.FloatTensor(x))


def _data_to_torch_dataset(data, window_size):

    X, G = data['X'], data['G']
    X = {key: WindowedData(val, window_size) for key, val in X.items()}
    G = {key: WindowedData(val, window_size) for key, val in G.items()}


    prognostics = [X[key] for key in prognostic_variables]
    forcings  = [G[key] for key in forcing_variables]
    return ConcatDataset(ConcatDataset(*prognostics), ConcatDataset(*forcings))


def _data_to_scaler(data, cuda=False):
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


def _to_dict(x):
    return {'sl': x[..., :34], 'qt': x[..., 34:]}


def _from_dict(prog):
    return torch.cat((prog['sl'], prog['qt']), -1)


def _data_to_loss_feature_weights(data, cuda=True):
    def _f(args):
        w, scale = args
        return w / scale**2

    w = merge_with(_f, data['w'], data['scales'])
    w = valmap(_numpy_to_variable, w)

    if cuda:
        w = valmap(lambda x: x.cuda(), w)

    return w


def _euler_step(prog, src, h):
    for key in prog:
        x = prog[key]
        f = src[key]
        prog = assoc(prog, key, x + h * f)
    return prog


def _scale_var(scale, mean, x):
    x = x.double()
    mu = mean.double()
    sig = scale.double()

    x = x.sub(mu)
    x = x.div(sig + 1e-7)

    return x.float()


@curry
def scaler(scales, means, x):
    out = {}
    for key in x:
        if key in scales and key in means:
            out[key] = _scale_var(scales[key], means[key], x[key])
        else:
            out[key] = x[key]
    return out


@curry
def weighted_loss(weight, x, y):
    return torch.mean(torch.pow(x - y, 2).mul(weight.float()))


def mlp(layer_sizes):
    layers = []
    n = len(layer_sizes)
    for k in range(n - 1):
        layers.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))
        if k < n - 2:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class ForcedStepper(nn.Module):
    def __init__(self, rhs, h, nsteps):
        super(ForcedStepper, self).__init__()
        self.nsteps = nsteps
        self.h = h
        self.rhs = rhs

    def forward(self, data: dict):
        """

        Parameters
        ----------
        data : dict
            A dictionary containing the prognostic variables and forcing data.
        """
        prog = data['prognostic']
        force = data['forcing']

        window_size = first(prog.values()).size(0)


        prog = valmap(lambda prog: prog[0], prog)
        # trapezoid rule
        force_dict = valmap(lambda prog: (prog[1:] + prog[:-1]) / 2, force)

        h = self.h
        nsteps = self.nsteps

        # output array
        steps = {key: [prog[key]] for key in prog}

        for i in range(1, window_size):
            for j in range(nsteps):
                src = self.rhs(prog)
                large_scale_forcing = {
                    key: val[i - 1]
                    for key, val in force_dict.items()
                }
                prog = _euler_step(prog, src, h / nsteps)
                prog = _euler_step(prog, large_scale_forcing, h / nsteps)

            # store data
            for key in prog:
                steps[key].append(prog[key])

        y = valmap(torch.stack, steps)
        return {'prognostic': y}


class RHS(nn.Module):
    def __init__(self, m, hidden=(), scaler=None):
        super(RHS, self).__init__()
        self.mlp = mlp((m, ) + hidden + (m, ))
        self.lin = nn.Linear(m, m, bias=False)
        self.scaler = scaler

    def forward(self, x):
        x = self.scaler(x)
        x = _from_dict(x)

        y = self.mlp(x) + self.lin(x)

        return _to_dict(y)


def _prepare_vars_in_nested_dict(data, cuda=False):
    if isinstance(data, torch.FloatTensor) or isinstance(data, torch.DoubleTensor):
        x =  Variable(data).float()
        if cuda:
            x = x.cuda()
        return x.transpose(0, 1)
    elif isinstance(data, dict):
        return {key: _prepare_vars_in_nested_dict(val, cuda=cuda)
                for key, val in data.items()}

def train_multistep_objective(data,
                              num_epochs=4,
                              window_size=10,
                              num_steps=500,
                              batch_size=100,
                              lr=0.01,
                              weight_decay=0.0,
                              nsteps=1,
                              nhidden=(10, 10, 10),
                              cuda=False,
                              test_loss=False):
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
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    scaler = _data_to_scaler(data, cuda=cuda)
    weights = _data_to_loss_feature_weights(data, cuda=cuda)

    # define the neural network
    m = sum(valmap(lambda x: x.size(-1), weights).values())

    rhs = RHS(m, hidden=nhidden, scaler=scaler)
    nstepper = ForcedStepper(rhs, h=dt, nsteps=nsteps)
    optimizer = torch.optim.Adam(
        rhs.parameters(), lr=lr, weight_decay=weight_decay)

    if cuda:
        nstepper.cuda()

    def loss(truth, pred):
        x = truth['prognostic']
        y = pred['prognostic']

        total_loss = 0
        for key in y:
            total_loss += weighted_loss(weights[key], x[key], y[key]) / len(y)
        return total_loss

    # _init_linear_weights(net, .01/nsteps)
    def closure(*args):
        # args = [Variable(x.float()).transpose(0, 1) for x in args]

        # if cuda:
        #     args = [arg.cuda() for arg in args]

        prog, force = args
        data = {
            'prognostic': dict(zip(prognostic_variables, prog)),
            'forcing': dict(zip(forcing_variables, force))
        }
        data = _prepare_vars_in_nested_dict(data)

        y = nstepper(data)
        return loss(data, y)

    # train the model
    if test_loss:
        args = next(iter(data_loader))
        return nstepper, closure(*args)
    else:
        train(
            data_loader,
            closure,
            optimizer=optimizer,
            num_epochs=num_epochs,
            num_steps=num_steps)
        return nstepper
