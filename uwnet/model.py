import torch
from collections import OrderedDict
from toolz import get, pipe, merge, first
from torch import nn

from uwnet.normalization import scaler
from uwnet import utils, constraints


def cat(seq):
    seq_with_nulldim = []
    sizes = []

    for x in seq:
        seq_with_nulldim.append(x)
        sizes.append(x.size(-1))

    return sizes, torch.cat(seq_with_nulldim, -1)


def uncat(sizes, x):
    return x.split(sizes, dim=-1)


class StackerScalerMixin(object):
    """Mixin object for translating fields between stacked array objects and
    dictionarys of arrays

    """

    def _stacked(self, x):
        return cat(get(list(self.input_fields), x))[1]

    def _unstacked(self, out):
        out = out.split(tuple(self.output_fields.values()), dim=-1)
        return dict(zip(self.output_fields.keys(), out))

    def _unscale(self, out):
        # scale the outputs
        for key in self.output_fields:
            if key in self.scale:
                out[key] = out[key] * self.scale[key].float()
        return out


class SaverMixin(object):
    """Mixin for output and initializing models from dictionaries of the state and
    arguments"""

    def to_dict(self):
        return {'args': self.args, 'kwargs': self.kwargs, 'state':
                self.state_dict()}

    @classmethod
    def from_dict(cls, d):
        mod = cls(*d['args'], **d['kwargs'])
        mod.load_state_dict(d['state'])
        return mod


class MOE(nn.Module):

    def __init__(self, m, n, n_experts):
        "docstring"
        super(MOE, self).__init__()

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(m, n),
                )
            for _ in range(n_experts)
        ])

        self.decider = nn.Sequential(
            nn.Linear(m, 256),
            nn.ReLU(),
            nn.Linear(256, n_experts),
            nn.Softmax(dim=1)
        )

    def poll_experts(self, x):
        return [expert(x) for expert in self.experts]

    def forward(self, x):
        weights = self.decider(x)
        ans = 0
        for val, w in zip(self.poll_experts(x),
                          weights.split(1, dim=-1)):
            ans = ans + w * val
        return ans


class MLP(nn.Module, StackerScalerMixin, SaverMixin):

    def __init__(self, mean, scale, time_step,
                 inputs=(('LHF', 1), ('SHF', 1), ('SOLIN', 1), ('qt', 34),
                         ('sl', 34), ('FQT', 34), ('FSL', 34)),
                 outputs=(('sl', 34), ('qt', 34))):

        "docstring"
        super(MLP, self).__init__()

        self.kwargs = {}
        self.kwargs['inputs'] = inputs
        self.kwargs['outputs'] = outputs

        n_in = sum(x[1] for x in inputs)
        n_out = sum(x[1] for x in outputs)

        # self.mod = MOE(n_in, n_out, n_experts=40)
        self.mod = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_out),
        )
        self.mean = mean
        self.scale = scale
        self.time_step = time_step
        self.scaler = scaler(scale, mean)

    def init_hidden(self, *args, **kwargs):
        return None

    @property
    def input_fields(self):
        return OrderedDict(self.kwargs['inputs'])

    @property
    def output_fields(self):
        return OrderedDict(self.kwargs['outputs'])

    @property
    def progs(self):
        """Prognostic variables are shared between the input and output
        sets
        """
        return set(self.output_fields) & set(self.input_fields)

    @property
    def diags(self):
        return set(self.output_fields.keys()) - set(self.input_fields)

    @property
    def aux(self):
        return set(self.input_fields) - set(self.output_fields.keys())

    @property
    def args(self):
        return (self.mean, self.scale, self.time_step)

    def rhs(self, x):
        """Estimated source terms and diagnostics"""
        x = {key: val if val.dim() == 2 else val.unsqueeze(-1)
             for key, val in x.items()}

        # zero out FSL in upper level
        # TODO this needs to be refactored
        # This routine should not know about FSL
        if 'FSL' in x:
            x['FSL'][:,-1] = 0.0

        stacked = pipe(x, self.scaler, self._stacked)
        out = self.mod(stacked)

        out = pipe(out, self._unstacked)

        sources = {key: out[key] for key in self.progs}
        diags = {key: out[key] for key in self.diags}

        if 'Prec' in self.diags:
            diags['Prec'].clamp_(0.0)

        return sources, diags

    def step(self, x, dt, *args):
        """Perform one time step using the neural network

        Parameters
        ----------
        x : dict
            dict of torch arrays (input variables)
        *args
            not used 

        Returns
        -------
        out : dict of torch arrays
            dict of the predictands. Either the next time step for the
            prognostic variables, or the predicted value for the
            non-prognostics (e.g. LHF, SHF, Prec).
        None
            placeholder to work with legacy code.

        """

        sources, diagnostics = self.rhs(x)

        out = {}
        for key in sources:
            out[key] = x[key] + dt * sources[key]

        out = merge(out, diagnostics)
        out = constraints.apply_constraints(x, out, dt)

        #  diagnostics
        if 'FQT' in self.input_fields:
            out['Q2NN'] = (out['qt'] - x['qt'])/dt - x['FQT']
            out['Q1NN'] = (out['sl'] - x['sl'])/dt - x['FSL']

        return out, None

    def forward(self, x, n=None):
        """
        Parameters
        ----------
        x
            variables. predictions will be made for all time points not
            included in the prognostic varibles in  x.
        n : int
            number of time steps of prognostic varibles to use before starting
            prediction

        Returns
        -------
        outputs
            dictionary of output_fields variables including prognostics and
            diagnostics

        """
        dt = torch.tensor(self.time_step, requires_grad=False)
        nt = x[first(self.input_fields.keys())].size(1)
        if n is None:
            n = nt
        output_fields = []

        aux = {key: x[key] for key in self.aux}

        for t in range(0, nt):
            if t < n:
                progs = {key: x[key][:, t] for key in self.progs}

            inputs = merge(utils.select_time(aux, t), progs)
            # This hardcoded string will cause trouble
            # This class should not know about layer_mass
            inputs['layer_mass'] = x['layer_mass']

            out, _ = self.step(inputs, dt)
            progs = {key: out[key] for key in self.progs}

            output_fields.append(out)

        return utils.stack_dicts(output_fields)

    def __repr__(self):
        return f"MLP({self.input_fields}, {self.output_fields})"
