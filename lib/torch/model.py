"""Torch module for time stepping and source term estimation. This module
contains classes for performing time stepping with prescribed forcing. It also
contains a class for the Radiation and source term estimation.

"""
import logging
from collections import defaultdict

import torch
from toolz import assoc, first, valmap
from torch import nn

from .. import constants

logger = logging.getLogger(__name__)


def _to_dict(x):
    return {
        'sl': x[..., :34],
        'qt': x[..., 34:]
    }


def _from_dict(prog):
    return torch.cat((prog['sl'], prog['qt']), -1)


def _euler_step(prog, src, h):
    for key in prog:
        x = prog[key]
        f = src[key]
        prog = assoc(prog, key, x + h * f)
    return prog


def padded_deriv(f, z):

    df = torch.zeros_like(f)

    df[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (z[2:] - z[:-2])
    df[..., 0] = (f[..., 1] - f[..., 0]) / (z[1] - z[0])
    df[..., -1] = (f[..., -1] - f[..., -2]) / (z[-1] - z[-2])

    return df


def vertical_advection(w, f, z):
    df = padded_deriv(f, z)
    return df * w * 86400


def large_scale_forcing(i, data):
    forcing = {
        key: (val[i - 1] + val[i]) / 2
        for key, val in data['forcing'].items()
    }
    return forcing


def compute_total_moisture(prog, data):
    w = data['constant']['w']
    return (prog['qt'] * w).sum(-1, keepdim=True)/1000




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


def where(cond, x, y):
    cond = cond.float()
    return cond * x + (1.0-cond) * y


def _fix_moisture_tend(q, fq, eps=1e-9, h=.125):
    cond = q + h * fq > eps
    return where(cond, fq, (eps-q)/h)


def mlp(layer_sizes):
    layers = []
    n = len(layer_sizes)
    for k in range(n - 1):
        layers.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))
        if k < n - 2:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


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

        progs = x
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

        # assure that q will remain positive after a step
        src['qt'] = _fix_moisture_tend(progs['qt'], src['qt'])

        return src, diags


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

                lsf = large_scale_forcing(i, data)

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


    @staticmethod
    def load_from_saved(d):
        from .data import scaler
        m, rhs_kw = d.pop('rhs')
        rhs_kw['scaler'] =  scaler(*rhs_kw.pop('scaler_args'))
        rhs = RHS(m, **rhs_kw)


        stepper_kw = d.pop('stepper')
        stepper = ForcedStepper(rhs, **stepper_kw)
        stepper.load_state_dict(d.pop('state'))

        return stepper


    def to_saved(self):

        m = self.rhs.lin.out_features
        nhidden =  self.rhs.mlp[0].out_features
        rhs_kwargs = dict(num_2d_inputs=self.rhs.lin.in_features - m,
                        hidden=(nhidden,),
                        precip_positive=self.rhs.precip_positive,
                        radiation=self.rhs.radiation,
                        scaler_args=self.rhs.scaler.args)
        output_dict = {
            'rhs': (m, rhs_kwargs),
            'stepper': dict(h=self.h, nsteps=self.nsteps),
            'state': self.state_dict()
        }

        return output_dict
