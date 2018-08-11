from collections import OrderedDict

import attr
from toolz import first, get, merge, pipe

import torch
from torch import nn
from uwnet import constraints, utils
from uwnet.normalization import scaler


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
    """Mixin for output and initializing models from dictionaries of the state
    and arguments

    Attributes
    ----------
    args
    kwargs
    """

    def to_dict(self):
        return {
            'args': self.args,
            'kwargs': self.kwargs,
            'state': self.state_dict()
        }

    @classmethod
    def from_dict(cls, d):
        mod = cls(*d['args'], **d['kwargs'])
        mod.load_state_dict(d['state'])
        return mod


class MOE(nn.Module):
    def __init__(self, m, n, n_experts):
        "docstring"
        super(MOE, self).__init__()

        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(m, n), ) for _ in range(n_experts)])

        self.decider = nn.Sequential(
            nn.Linear(m, 256),
            nn.ReLU(),
            nn.Linear(256, n_experts),
            nn.Softmax(dim=1))

    def poll_experts(self, x):
        return [expert(x) for expert in self.experts]

    def forward(self, x):
        weights = self.decider(x)
        ans = 0
        for val, w in zip(self.poll_experts(x), weights.split(1, dim=-1)):
            ans = ans + w * val
        return ans


class AbstractApparentSource(nn.Module):
    """Class for computing the apparent sources of variables

    Attributes
    ----------
    mean : dict
        dict of means to use for centering variables
    scales : dict
        dict of scales to use for scaling variables
    """

    def forward(self, h, aux, prog):
        """Compute the apparent tendencies of the prognostic variables

        Parameters
        ----------
        h : float
            time step
        aux : dict
            dict of auxiliary inputs
        prog : dict
             dict of prognostic variables

        Returns
        -------
        sources : dict
            apparent sources of prognostic variables
        diags : dict
            diagnostic quantities (e.g. precipitation, latent heat flux, etc)
        """
        raise NotImplementedError


@attr.s
class VariableSpec(object):
    """Specification data for variable inputs

    This class allows passing relevant metadata such as physical units,
    dimension, and name.
    """
    name = attr.ib()
    num = attr.ib()
    positive = attr.ib(default=False)
    conserved = attr.ib(default=False)
    units = attr.ib(default='')


@attr.s
class VariableList(object):
    """List of VariableSpec objects"""
    variables = attr.ib()

    @property
    def num(self):
        if len(self.variables) > 0:
            return sum(x.num for x in self.variables)
        else:
            return 0

    def to_dict(self):
        return OrderedDict([(var.name, var.num) for var in self.variables])

    @classmethod
    def from_tuples(cls, inputs):
        return cls([VariableSpec(*input) for input in inputs])

    def __getitem__(self, key):
        return self.variables[0]

    def __iter__(self):
        return iter(self.variables)


class MLP(nn.Module, StackerScalerMixin, SaverMixin):
    def __init__(self,
                 mean,
                 scale,
                 time_step,
                 inputs=(('LHF', 1), ('SHF', 1), ('SOLIN', 1), ('qt', 34),
                         ('sl', 34), ('FQT', 34), ('FSL', 34)),
                 outputs=(('sl', 34), ('qt', 34))):

        "docstring"
        super(MLP, self).__init__()

        self.kwargs = {}
        self.kwargs['inputs'] = inputs
        self.kwargs['outputs'] = outputs

        self.inputs = VariableList.from_tuples(inputs)
        self.outputs = VariableList.from_tuples(outputs)

        n_in = self.inputs.num
        n_out = self.outputs.num

        # self.mod = MOE(n_in, n_out, n_experts=40)
        self.mod = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_out), )
        self.mean = mean
        self.scale = scale
        self.time_step = time_step
        self.scaler = scaler(scale, mean)

    def init_hidden(self, *args, **kwargs):
        return None

    @property
    def input_fields(self):
        return self.inputs.to_dict()

    @property
    def output_fields(self):
        return self.outputs.to_dict()

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

    def rhs(self, aux, progs):
        """Estimated source terms and diagnostics"""
        x = {}
        x.update(aux)
        x.update(progs)

        x = {
            key: val if val.dim() == 2 else val.unsqueeze(-1)
            for key, val in x.items()
        }

        # zero out FSL in upper level
        # TODO this needs to be refactored
        # This routine should not know about FSL
        if 'FSL' in x:
            x['FSL'][:, -1] = 0.0

        stacked = pipe(x, self.scaler, self._stacked)
        out = self.mod(stacked)

        out = pipe(out, self._unstacked)

        sources = {key: out[key] for key in progs}
        diags = {key: val for key, val in out.items() if key not in progs}
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

        aux = {key: val for key, val in x.items() if key in self.aux}
        progs = {key: val for key, val in x.items() if key in self.progs}

        sources, diagnostics = self.rhs(aux, progs)

        out = {}
        for key in sources:
            out[key] = x[key] + dt * sources[key]

        out = merge(out, diagnostics)
        out = constraints.apply_constraints(x, out, dt, output_specs=self.outputs)

        #  diagnostics
        if 'FQT' in self.input_fields:
            out['Q2NN'] = (out['qt'] - x['qt']) / dt - x['FQT']
            out['Q1NN'] = (out['sl'] - x['sl']) / dt - x['FSL']

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
