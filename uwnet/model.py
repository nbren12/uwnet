from collections import OrderedDict

import attr
from toolz import first, get, merge, pipe, assoc, merge_with

import torch
from torch import nn
from uwnet import constraints, utils
from uwnet.normalization import scaler
from uwnet.modules import LinearDictIn, LinearDictOut


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

    @classmethod
    def from_path(cls, path):
        d = torch.load(path)['dict']
        return cls.from_dict(d)


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
        for var in self.variables:
            n = x[var.name].size(-1)
            if n != var.num:
                raise ValueError(
                    f"{var.name} has {n} features. Expected {var.num}")
        return cat(get(self.names, x))[1]

    def unstack(self, x):
        x = x.split(tuple(self.nums), dim=-1)
        return dict(zip(self.names, x))


class MLP(nn.Module, SaverMixin):
    """
    Attributes
    ----------
    add_forcing : bool
        If true, FQT, FU, etc in the inputs to MLP.forward will be added to the
        produce the final tendencies.
    forcings
        List of tuples of the form (prognostic_var_name, num), where
        prognostic_var_name is one of the inputs. The input batch dictionary
        should contain the key 'F<prognostic_var_name>'
    """

    def __init__(self,
                 mean,
                 scale,
                 time_step,
                 inputs=(('LHF', 1), ('SHF', 1), ('SOLIN', 1), ('QT', 34),
                         ('SLI', 34), ('FQT', 34), ('FSLI', 34)),
                 outputs=(('SLI', 34), ('QT', 34)),
                 forcings=(),
                 add_forcing=False,
                 **kwargs):

        "docstring"
        super(MLP, self).__init__()

        self.kwargs = locals()
        self.kwargs.pop('self')
        self.kwargs.pop('__class__')

        self.forcings = forcings
        self.add_forcing = add_forcing
        self.inputs = VariableList.from_tuples(inputs)
        self.outputs = VariableList.from_tuples(outputs)

        self.mean = mean
        self.scale = scale
        self.time_step = time_step
        self.scaler = scaler(scale, mean)

        self.mod = nn.Sequential(
            LinearDictIn([(x.name, x.num) for x in self.inputs], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            LinearDictOut(256, [(x.name, x.num) for x in self.outputs]), )

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
        return (set(self.inputs.names) - set(self.outputs.names))\
            .union({'F' + key for key, num in self.forcings})

    @property
    def args(self):
        return (self.mean, self.scale, self.time_step)

    def rhs(self, aux, progs):
        """Estimated source terms and diagnostics

        All inputs have shape (*, z, y, x) or (*, y, x)

        """
        z_axis = -3
        x = {}
        x.update(aux)
        x.update(progs)

        # make the inputs have the right shape
        inputs = {}
        for spec in self.inputs:
            val = x[spec.name]
            if spec.num == 1:
                val = val.unsqueeze(z_axis)

            val = val.transpose(z_axis, -1)
            inputs[spec.name] = val

        scaled = self.scaler(inputs)
        out = self.mod(scaled)

        # make the outputs have the right shape
        for spec in self.outputs:
            if spec.num == 1:
                out[spec.name] = out[spec.name].squeeze(-1)
            else:
                out[spec.name] = out[spec.name].transpose(z_axis, -1)

        # scale the derivative terms
        sources = {}
        for key in progs:
            sources[key] = out[key] / 86400

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
            forcing_key = 'F' + key
            nn_forcing_key = 'F' + key + 'NN'
            if self.add_forcing:
                out[key] = x[key] + dt * (x[forcing_key] + sources[key])
            else:
                out[key] = x[key] + dt * sources[key]

            x = assoc(x, forcing_key, 0.0)
            # store the NN forcing as a diagnostic
            diagnostics[nn_forcing_key] = sources[key]

        out = merge(out, diagnostics)
        return out

    def train(self, val=True):
        super(MLP, self).train()
        self.add_forcing = val

    def forward(self, x, n=None, dt=None):
        """
        Parameters
        ----------
        x
            variables. predictions will be made for all time points not
            included in the prognostic varibles in  x.
        n : int
            number of time steps of prognostic varibles to use before starting
            prediction
        dt : float
           time step, defaults to self.time_step

        Returns
        -------
        outputs
            dictionary of output_fields variables including prognostics and
            diagnostics

        """
        if dt is None:
            dt = torch.tensor(self.time_step, requires_grad=False).float()
        else:
            dt = torch.tensor(dt, requires_grad=False).float()

        nt = x[first(self.inputs.names)].size(0)
        if n is None:
            n = nt
        output_fields = []

        aux = {key: x[key] for key in self.aux}

        for t in range(0, nt):

            # make the correct inputs
            if t < n:
                progs = {key: x[key][t] for key in self.progs}

            aux_t = {key: val[t] for key, val in aux.items()}
            inputs = merge(aux_t, progs)

            # Call the network
            out = self.step(inputs, dt)

            # save outputs for the next step
            progs = {key: out[key] for key in self.progs}
            output_fields.append(out)

        # stack the outputs
        return merge_with(torch.stack, output_fields)

    @property
    def z(self):
        nz = max(spec.num for spec in self.inputs)
        return range(nz)

    def call_with_xr(self, ds, **kwargs):
        """Call the neural network with xarray inputs"""
        import xarray as xr
        d = {key: torch.from_numpy(ds[key].values) for key in ds.data_vars}
        with torch.no_grad():
            output = self(d, **kwargs)

        # parse output
        data_vars = {}
        for key, val in output.items():
            if val.size(1) == 1:
                data_vars[key] = (['time', 'y', 'x'],
                                  output[key].numpy().squeeze())
            else:
                data_vars[key] = (['time', 'z', 'y', 'x'],
                                  output[key].numpy())

        return xr.Dataset(data_vars, coords=ds.coords)

    def call_with_numpy_dict(self, inputs, **kwargs):
        """Call the neural network with numpy inputs"""

        with torch.no_grad():
            d = {key: torch.from_numpy(val) for key, val in inputs.items()}
            output = self(d, **kwargs)
            return {key: val.numpy() for key, val in output.items()}

    def __repr__(self):
        return f"MLP({self.inputs.names}, {self.outputs.names})"
