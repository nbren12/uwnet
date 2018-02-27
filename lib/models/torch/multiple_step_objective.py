"""Fit model for the multiple time step objective function. This requires some
special dataloaders etc

"""
from collections import defaultdict
from toolz import curry, first, valmap, merge_with, assoc
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .datasets import DictDataset, WindowedData
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
                    time_slice=None,
                    prognostic_variables=('sl', 'qt'),
                    forcing_variables=('sl', 'qt', 'QRAD', 'LHF', 'SHF',
                                       'Prec', 'W', 'SOLIN')):

    if not time_slice:
        time_slice = slice(None)

    X, G = data['X'], data['G']
    X = DictDataset({
        key: WindowedData(X[key][time_slice], window_size)
        for key in prognostic_variables
    })
    G = DictDataset(
        {key: WindowedData(G[key][time_slice], window_size)
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
    return {
        'sl': x[..., :34],
        'qt': x[..., 34:]
    }


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
    # return torch.mean(torch.pow(x - y, 2).mul(weight.float()))
    return torch.mean(torch.abs(x - y).mul(weight.float()))


def mlp(layer_sizes):
    layers = []
    n = len(layer_sizes)
    for k in range(n - 1):
        layers.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))
        if k < n - 2:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def padded_deriv(f, z):

    df = torch.zeros_like(f)

    df[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (z[2:] - z[:-2])
    df[..., 0] = (f[..., 1] - f[..., 0]) / (z[1] - z[0])
    df[..., -1] = (f[..., -1] - f[..., -2]) / (z[-1] - z[-2])

    return df


def vertical_advection(w, f, z):
    df = padded_deriv(f, z)
    return df * w * 86400


def large_scale_forcing(i, prog, data):
    forcing = {
        key: (val[i - 1] + val[i]) / 2
        for key, val in data['forcing'].items()
    }

    for key, val in prog.items():
        z = data['constant']['z']
        vert_adv = vertical_advection(forcing['W'], val, z)
        forcing[key] = forcing[key] - vert_adv

    return forcing


def compute_total_moisture(prog, data):
    w = data['constant']['w']
    return (prog['qt'] * w).sum(-1, keepdim=True)/1000


class ForcedStepper(nn.Module):
    def __init__(self, rhs, h, nsteps, interactive_vertical_adv=False):
        super(ForcedStepper, self).__init__()
        self.nsteps = nsteps
        self.h = h
        self.rhs = rhs
        self.interactive_vertical_adv = interactive_vertical_adv

    def forward(self, data: dict):
        """

        Parameters
        ----------
        data : dict
            A dictionary containing the prognostic variables and forcing data.
        """
        data = data.copy()
        prog = data['prognostic']

        window_size = first(prog.values()).size(0)
        prog = valmap(lambda prog: prog[0], prog)
        h = self.h
        nsteps = self.nsteps

        # output array
        steps = {key: [prog[key]] for key in prog}
        # diagnostics
        diagnostics = defaultdict(list)

        for i in range(1, window_size):
            diag_step = defaultdict(lambda: 0)
            for j in range(nsteps):

                if self.interactive_vertical_adv:
                    lsf = large_scale_forcing(i, prog, data)
                else:
                    lsf_prog = valmap(lambda x: (x[i - 1] + x[i]) / 2,
                                      data['prognostic'])
                    lsf = large_scale_forcing(i, lsf_prog, data)

                # apply large scale forcings
                prog = _euler_step(prog, lsf, h / nsteps)
                total_moisture_before = compute_total_moisture(prog, data)

                # compute and apply rhs using neural network
                src, diags = self.rhs(prog, lsf, data['constant']['w'])

                prog = _euler_step(prog, src, h / nsteps)
                total_moisture_after = compute_total_moisture(prog, data)
                evap = lsf['LHF'] * 86400 / 2.51e6 * h/nsteps
                prec = (total_moisture_before - total_moisture_after + evap) / h * nsteps
                diags['Prec'] = prec

                # running average of diagnostics
                for key in diags:
                    diag_step[key] = diag_step[key] + diags[key] / nsteps

            # store accumulated diagnostics
            for key in diag_step:
                diagnostics[key].append(diag_step[key])

            # store data
            for key in prog:
                steps[key].append(prog[key])

        y = data.copy()
        y['prognostic'] = valmap(torch.stack, steps)
        y['diagnostic'] = valmap(torch.stack, diagnostics)
        return y


def precip_from_s(fsl, qrad, shf, w):
    return (w * (fsl - qrad)).sum(
        -1, keepdim=True
    ) * constants.cp / constants.Lv - shf * 86400 / constants.Lv


def precip_from_q(fqt, lhf, w):
    return -(fqt * w).sum(
        -1, keepdim=True) / 1000. + lhf * 86400 / constants.Lv


def mass_integrate(x, w):
    return (x * w).sum(-1, keepdim=True)


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
    f_col_target = qrad_col + (
        shf * 86400 + constants.Lv * precip) / constants.cp

    return fsl - f_col / mass + f_col_target / mass


def enforce_precip_qt(fqt, lhf, precip, w):
    """Adjust moisture tendency to match a target precipitation

    .. math::

        Lv < f >  = LHF - L P

    Parameters
    ----------
    fqt : mm/day
    lhf : W/m^2
    precip : mm/day
    w : kg /m^2

    """

    mass = mass_integrate(1.0, w)
    f_col = mass_integrate(fqt, w)
    f_col_target = (lhf * 86400 / constants.Lv - precip) * 1000

    return fqt - f_col / mass + f_col_target / mass


class Qrad(nn.Module):
    def __init__(self):
        super(Qrad, self).__init__()
        n = 71
        m = 34

        nhid = 128

        self.net = nn.Sequential(
            nn.BatchNorm1d(n),
            nn.Linear(n, nhid),
            nn.ReLU(),
            nn.BatchNorm1d(nhid),
            nn.Linear(nhid, nhid),
            nn.ReLU(), nn.BatchNorm1d(nhid), nn.Linear(nhid, m))
        self.n = n
        self.m = m

    def forward(self, x):
        return self.net(x)


class RHS(nn.Module):
    def __init__(self,
                 m,
                 hidden=(),
                 scaler=None,
                 num_2d_inputs=3,
                 precip_positive=True,
                 radiation='interactive'):
        """
        Parameters
        ----------
        radiation : str
            'interactive', 'prescribed', or 'zero'.
        precip_positive : bool
            constrain precip to be positive if True
        """
        super(RHS, self).__init__()
        self.mlp = mlp((m + num_2d_inputs, ) + hidden + (m, ))
        self.lin = nn.Linear(m + num_2d_inputs, m, bias=False)
        self.scaler = scaler
        self.bn = nn.BatchNorm1d(num_2d_inputs)
        self.qrad = Qrad()
        self.radiation = radiation
        self.precip_positive = precip_positive

    def forward(self, x, force, w):
        diags = {}
        x = self.scaler(x)
        f = self.scaler(force)

        data_2d = torch.cat((f['SHF'], f['LHF'], f['SOLIN']), -1)
        data_2d = self.bn(data_2d)

        x = _from_dict(x)
        x = torch.cat((x, data_2d), -1)
        y = self.mlp(x) + self.lin(x)
        src = _to_dict(y)


        # compute radiation
        if self.radiation == 'interactive':
            qrad = self.qrad(x)
            diags['QRAD'] = qrad
            src['sl'] = src['sl'] + qrad
        elif self.radiation == 'prescribed':
            qrad = force['QRAD']
            src['sl'] = src['sl'] + qrad
        elif self.radiation == 'zero':
            qrad = force['QRAD']

        # Compute the precipitation from q
        PrecT = precip_from_s(src['sl'], qrad, force['SHF'], w)
        PrecQ = precip_from_q(src['qt'], force['LHF'], w)
        Prec = (PrecT + PrecQ) / 2.0
        diags['Prec'] = Prec
        if self.precip_positive:
            # precip must be > 0
            Prec = Prec.clamp(0.0)
            # ensure that sl and qt budgets estimate the same precipitation
            src = {
                'sl': enforce_precip_sl(src['sl'], qrad, force['SHF'], Prec,
                                        w),
                'qt': enforce_precip_qt(src['qt'], force['LHF'], Prec, w)
            }  # yapf: disable

        return src, diags


def train_multistep_objective(data, num_epochs=4, window_size=10,
                              num_test_examples=10000,
                              test_window_size=100,
                              num_batches=500, batch_size=100, lr=0.01,
                              weight_decay=0.0, nsteps=1, nhidden=(10, 10, 10),
                              cuda=False, test_loss=False,
                              precip_in_loss=False, precip_positive=True,
                              radiation='zero',
                              interactive_vertical_adv=False):
    """Train a single layer perceptron euler time stepping model

    For one time step this torch models performs the following math

    .. math::

        x^n+1 = x^n + dt * (f(x^n) + g)


    """

    arguments = locals()
    arguments.pop('data')

    torch.manual_seed(1)

    # the sampling interval of the data
    dt = Variable(torch.FloatTensor([3 / 24]), requires_grad=False)
    if cuda:
        dt = dt.cuda()

    train_slice = slice(0, 325)
    test_slice = slice(325, None)

    train_dataset = prepare_dataset(data, window_size, time_slice=train_slice)
    test_dataset = prepare_dataset(data, test_window_size,
                                   time_slice=test_slice)

    # train on only a bootstrap sample
    training_inds = np.random.choice(len(train_dataset),
                                     num_batches * batch_size, replace=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=SubsetRandomSampler(training_inds))

    # make a test_loader object
    testing_inds = np.random.choice(len(test_dataset), num_test_examples,
                                    replace=False)
    test_loader = DataLoader(test_dataset, batch_size=len(testing_inds),
                             sampler=SubsetRandomSampler(testing_inds))

    scaler = _data_to_scaler(data, cuda=cuda)
    weights = _data_to_loss_feature_weights(data, cuda=cuda)

    # define the neural network
    m = sum(valmap(lambda x: x.size(-1), weights).values())

    rhs = RHS(
        m,
        hidden=nhidden,
        scaler=scaler,
        radiation=radiation,
        precip_positive=precip_positive)

    nstepper = ForcedStepper(
        rhs,
        h=dt,
        nsteps=nsteps,
        interactive_vertical_adv=interactive_vertical_adv)

    optimizer = torch.optim.Adam(
        rhs.parameters(), lr=lr, weight_decay=weight_decay)

    constants = {
        'w': Variable(torch.FloatTensor(data['w']['sl'])),
        'z': Variable(torch.FloatTensor(data['z']))
    }
    if cuda:
        nstepper.cuda()
        for key in constants:
            constants[key] = constants[key].cuda()

    def loss(truth, pred):
        x = truth['prognostic']
        y = pred['prognostic']

        total_loss = 0
        # time series loss
        for key in y:
            total_loss += weighted_loss(weights[key], x[key], y[key]) / len(y)

        if precip_in_loss:
            # column budget losses this compares he predicted precipitation for
            # each field to reality
            prec = truth['forcing']['Prec']
            predicted = pred['diagnostic']['Prec']
            observed = (prec[1:] + prec[:-1]) / 2
            total_loss += torch.mean(torch.pow(observed - predicted, 2)) / 5

        if radiation == 'interactive':
            qrad = truth['forcing']['QRAD']
            predicted = pred['diagnostic']['QRAD'][0]
            observed = qrad[0]
            total_loss += torch.mean(torch.pow(observed - predicted, 2))

        return total_loss

    # _init_linear_weights(net, .01/nsteps)
    def closure(batch):
        batch = _prepare_vars_in_nested_dict(batch, cuda=cuda)
        batch['constant'] = constants
        y = nstepper(batch)
        return loss(batch, y)


    epoch_data = []
    def monitor(state):
        loss = sum(closure(batch) for batch in test_loader)
        avg_loss  = loss / len(test_loader)
        state['test_loss'] = float(avg_loss)
        epoch_data.append(state)

        print("Epoch: {epoch}; Test Loss [{test_loss}]; Train Loss[{train_loss}]".format(**state))


    # train the model
    if test_loss:
        args = next(iter(data_loader))
        return nstepper, closure(args)
    else:
        train(
            train_loader,
            closure,
            optimizer=optimizer,
            monitor=monitor,
            num_epochs=num_epochs)

        training_metadata = {
            'args': arguments,
            'training': epoch_data
        }
        return nstepper, training_metadata
