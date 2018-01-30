"""Fit model for the multiple time step objective function. This requires some special dataloaders etc

"""
from collections import defaultdict
from toolz import curry, first, valmap, merge_with, assoc
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .datasets import DictDataset, ConcatDataset, WindowedData
from .utils import train

from ... import constants


def _prepare_vars_in_nested_dict(data, cuda=False):
    if torch.is_tensor(data):
        x = Variable(data).float()
        if cuda:
            x = x.cuda()
        return x.transpose(0, 1)
    elif isinstance(data, dict):
        return {
            key: _prepare_vars_in_nested_dict(val, cuda=cuda)
            for key, val in data.items()
        }


def prepare_dataset(data,
                    window_size,
                    prognostic_variables=('sl', 'qt'),
                    forcing_variables=('sl', 'qt', 'QRAD', 'LHF', 'SHF',
                                       'Prec', 'W')):

    X, G = data['X'], data['G']
    X = DictDataset({
        key: WindowedData(X[key], window_size)
        for key in prognostic_variables
    })
    G = DictDataset(
        {key: WindowedData(G[key], window_size)
         for key in forcing_variables})

    return DictDataset({'prognostic': X, 'forcing': G})


def _numpy_to_variable(x):
    return Variable(torch.FloatTensor(x))


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
        data = data.copy()
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
                large_scale_forcing = {
                    key: val[i - 1]
                    for key, val in force_dict.items()
                }
                src = self.rhs(prog, large_scale_forcing, data['constant']['w'])
                prog = _euler_step(prog, src, h / nsteps)
                prog = _euler_step(prog, large_scale_forcing, h / nsteps)

            # store data
            for key in prog:
                steps[key].append(prog[key])

        y = data.copy()
        y['prognostic'] = valmap(torch.stack, steps)
        y['diagnostic'] = self._diagnostics(y, h)
        return y

    def _diagnostics(self, data, h):

        w = data['constant']['w']

        src = {
            key: (x[1:] - x[:-1]) / h
            for key, x in data['prognostic'].items()
        }

        f = valmap(lambda prog: (prog[1:] + prog[:-1]) / 2, data['forcing'])
        prec_t = precip_from_s(src['sl'], f['QRAD'], f['SHF'], w)
        prec_q = precip_from_q(src['qt'], f['LHF'], w)

        return {'prec_t': prec_t, 'prec_q': prec_q}


def precip_from_s(fsl, qrad, shf, w):
    return (w * (fsl - qrad)).sum(-1, keepdim=True) * constants.cp / constants.Lv - shf * 86400 / constants.Lv


def precip_from_q(fqt, lhf, w):
    return -(fqt * w).sum(-1, keepdim=True) / 1000. + lhf * 86400 / constants.Lv


def mass_integrate(x, w):
    return (x*w).sum(-1, keepdim=True)


def enforce_precip_sl(fsl, qrad, shf, precip, w):
    """Adjust temperature tendency to match a target precipitation

    .. math::

        cp < f >  = cp <QRAD> + SHF + L P

    Parameters
    ----------
    fsl : K/day
    qrad : K/day
    shf : W/m^2
    precip : mm/day
    w : kg /m^2

    """

    mass = mass_integrate(1.0, w)
    qrad_col = mass_integrate(qrad, w)
    f_col = mass_integrate(fsl, w)
    f_col_target = qrad_col + (shf * 86400 + constants.Lv * precip)/constants.cp

    return fsl - f_col/mass + f_col_target/mass


def enforce_precip_qt(fqt, lhf, precip, w):
    """Adjust moisture tendency to match a target precipitation

    .. math::

    """


class RHS(nn.Module):
    def __init__(self, m, hidden=(), scaler=None):
        super(RHS, self).__init__()
        self.mlp = mlp((m, ) + hidden + (m, ))
        self.lin = nn.Linear(m, m, bias=False)
        self.scaler = scaler

        # network precipitation
        self.precip = nn.Sequential(
            nn.BatchNorm1d(68),
            nn.Linear(68, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 1),
            nn.ReLU()
        )

    def forward(self, x, force, w):
        x = self.scaler(x)
        x = _from_dict(x)

        y = self.mlp(x) + self.lin(x)

        src = _to_dict(y)

        fsl = enforce_precip_sl(src['sl'], force['QRAD'], force['SHF'], 1, w)


        from IPython import embed; embed()
        # P = self.precip(x)
        # sl_col = (w*src['sl']).sum(-1, keepdim=True)
        # qt_col = (w*src['qt']).sum(-1, keepdim=True)
        # rad_col = (w*force['QRAD']).sum(-1, keepdim=True)

        # sl_col_expected = constants.Lv/constants.cp * P - force['SHF'] * 86400/constants.cp - rad_col
        # qt_col_expected = 1000*(force['LHF'] * 86400/constants.Lv - P)


        # src['sl'] = src['sl'] - sl_col + sl_col_expected
        # src['qt'] = src['qt'] - qt_col + qt_col_expected
        # psl  = precip_from_s(src['sl'], force['QRAD'], force['SHF'], w)
        return src


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

    dataset = prepare_dataset(data, window_size)
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

        # time series loss
        total_loss = 0
        for key in y:
            total_loss += weighted_loss(weights[key], x[key], y[key]) / len(y)

        # column budget losses this compares the predicted precipitation for
        # each field to reality
        prect = pred['diagnostic']['prec_t']
        precq = pred['diagnostic']['prec_q']
        prec = truth['forcing']['Prec']
        prec = (prec[1:] + prec[:-1]) / 2

        total_loss += torch.mean(torch.pow(prect - prec, 2)) / 10
        total_loss += torch.mean(torch.pow(precq - prec, 2)) / 10

        return total_loss

    # _init_linear_weights(net, .01/nsteps)
    def closure(batch):
        batch = _prepare_vars_in_nested_dict(batch, cuda=cuda)
        batch['constant'] = {'w': Variable(torch.FloatTensor(data['w']['sl']))}
        y = nstepper(batch)
        return loss(batch, y)

    # train the model
    if test_loss:
        args = next(iter(data_loader))
        return nstepper, closure(args)
    else:
        train(
            data_loader,
            closure,
            optimizer=optimizer,
            num_epochs=num_epochs,
            num_steps=num_steps)
        return nstepper
