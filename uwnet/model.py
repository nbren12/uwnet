from collections import OrderedDict

import attr
from toolz import first, get, merge, pipe

import torch
from torch import nn
from uwnet import constraints, utils
from uwnet.normalization import scaler
import inspect


def cat(seq):
    seq_with_nulldim = []
    sizes = []

    for x in seq:
        seq_with_nulldim.append(x)
        sizes.append(x.size(-1))

    return sizes, torch.cat(seq_with_nulldim, -1)


def uncat(sizes, x):
    return x.split(sizes, dim=-1)


class SaverMixin(object):
    """Mixin for output and initializing models from dictionaries of the state
    and arguments

    Attributes
    ----------
    args
    kwargs
    """

    def to_dict(self):
        return {'kwargs': self.kwargs, 'state': self.state_dict()}

    @classmethod
    def from_dict(cls, d):
        mod = cls(**d['kwargs'])
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
        return self.variables[key]

    def __iter__(self):
        return iter(self.variables)

    @property
    def names(self):
        return [x.name for x in self.variables]

    @property
    def nums(self):
        return tuple(x.num for x in self.variables)

    def stack(self, x):
        return cat(get(self.names, x))[1]

    def unstack(self, x):
        x = x.split(tuple(self.nums), dim=-1)
        return dict(zip(self.names, x))


class MLP(nn.Module, SaverMixin):
    def __init__(self,
                 mean,
                 scale,
                 time_step,
                 inputs=(('LHF', 1), ('SHF', 1), ('SOLIN', 1), ('QT', 34),
                         ('SLI', 34), ('FQT', 34), ('FSLI', 34)),
                 outputs=(('SLI', 34), ('QT', 34)),
                 add_forcing=False, **kwargs):

        "docstring"
        super(MLP, self).__init__()

        self.kwargs = locals()
        self.kwargs.pop('self')
        self.kwargs.pop('__class__')

        self.add_forcing = add_forcing
        self.inputs = VariableList.from_tuples(inputs)
        self.outputs = VariableList.from_tuples(outputs)

        self.mean = mean
        self.scale = scale
        self.time_step = time_step
        self.scaler = scaler(scale, mean)

        self.mod = nn.Sequential(
            nn.Linear(self.inputs.num, 256),
            nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, self.outputs.num))

    def init_hidden(self, *args, **kwargs):
        return None

    @property
    def progs(self):
        """Prognostic variables are shared between the input and output
        sets
        """
        return set(self.outputs.names) & set(self.inputs.names)

    @property
    def diags(self):
        return set(self.outputs.names) - set(self.inputs.names)

    @property
    def aux(self):
        return set(self.inputs.names) - set(self.outputs.names)

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
        if 'FSLI' in x:
            x['FSLI'][:, -1] = 0.0

        stacked = pipe(x, self.scaler, self.inputs.stack)
        out = self.mod(stacked)

        out = pipe(out, self.outputs.unstack)

        # handle prognostic variables
        sources = {}
        for key in progs:
            src = out[key] / 86400
            if self.add_forcing:
                src += x['F' + key]
            sources[key] = src

        diags = {key: val for key, val in out.items() if key not in progs}
        return sources, diags

    def step(self, x, dt, *args):
        """Perform one time step using the neural network

        Parameters
        ----------
        x : dict
            dict of torch arrays (input variables)
        dt : float
            the timestep in seconds
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
        out = constraints.apply_constraints(
            x, out, dt, output_specs=self.outputs)

        #  diagnostics
        try:
            out['Q2NN'] = (out['QT'] - x['QT']) / dt - x['FQT']
            out['Q1NN'] = (out['SLI'] - x['SLI']) / dt - x['FSLI']
        except KeyError:
            pass

        return out

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
        nt = x[first(self.inputs.names)].size(1)
        if n is None:
            n = nt
        output_fields = []

        aux = {key: x[key] for key in set(x) - set(self.outputs.names)}

        for t in range(0, nt):
            if t < n:
                progs = {key: x[key][:, t] for key in self.progs}

            inputs = merge(utils.select_time(aux, t), progs)
            # This hardcoded string will cause trouble
            # This class should not know about layer_mass
            inputs['layer_mass'] = x['layer_mass']

            out = self.step(inputs, dt)
            progs = {key: out[key] for key in self.progs}

            output_fields.append(out)

        return utils.stack_dicts(output_fields)

    def __repr__(self):
        return f"MLP({self.inputs.names}, {self.outputs.names})"
