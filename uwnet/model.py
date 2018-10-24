from collections import OrderedDict

import attr
import torch
from toolz import first, get, merge, merge_with, pipe
from functools import partial
from torch import nn

import xarray as xr
from uwnet.modules import LinearDictIn, LinearDictOut
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


def _dataset_to_torch_dict(ds):
    return {key: torch.from_numpy(ds[key].values).float() for key in ds.data_vars}


def _torch_dict_to_dataset(output, coords):
    # parse output
    data_vars = {}

    dims_2d = [dim for dim in ['time', 'y', 'x'] if dim in coords.dims]
    dims_3d = [dim for dim in ['time', 'z', 'y', 'x'] if dim in coords.dims]

    time_dim_present = 'time' in dims_2d

    for key, val in output.items():
        if time_dim_present:
            nz = val.size(1)
        else:
            nz = val.size(0)

        if nz == 1:
            data_vars[key] = (dims_2d, output[key].numpy().squeeze())
        else:
            data_vars[key] = (dims_3d, output[key].numpy())

    # prepare coordinates
    coords = dict(coords.items())
    return xr.Dataset(data_vars, coords=coords)


def call_with_xr(self, ds, drop_times=1, **kwargs):
    """Call the neural network with xarray inputs"""
    tensordict = _dataset_to_torch_dict(ds)
    with torch.no_grad():
        output = self(tensordict, **kwargs)
    return _torch_dict_to_dataset(output, ds.coords, drop_times)


def _numpy_dict_to_torch_dict(inputs):
    return {key: torch.from_numpy(val) for key, val in inputs.items()}


def _torch_dict_to_numpy_dict(output):
    return {key: val.numpy() for key, val in output.items()}


def call_with_numpy_dict(self, inputs, **kwargs):
    """Call the neural network with numpy inputs"""
    with torch.no_grad():
        return pipe(inputs,
                    _numpy_dict_to_torch_dict,
                    partial(self, **kwargs),
                    _torch_dict_to_numpy_dict)


class SaverMixin(object):
    """Mixin for output and initializing models from dictionaries of the state
    and arguments

    Attributes
    ----------
    args
    kwargs

    Notes
    -----
    To easily get this to work with a new class add these lines at the top of
    the __init__ method::

        self.kwargs = locals()
        self.kwargs.pop('self')
        self.kwargs.pop('__class__')

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
    z_axis = attr.ib(default=-3)

    def move_height_to_last_dim(self, val):
        if self.num == 1:
            val = val.unsqueeze(self.z_axis)
        return val.transpose(self.z_axis, -1)


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


def _height_to_last_dim(x, input_specs, z_axis=-3):
    # make the inputs have the right shape
    return {spec.name: spec.move_height_to_last_dim(x[spec.name])
            for spec in input_specs}


def _height_to_original_dim(out, specs, z_axis=-3):
    # make the outputs have the right shape
    for spec in specs:
        if spec.num == 1:
            out[spec.name] = out[spec.name].squeeze(-1)
        else:
            out[spec.name] = out[spec.name].transpose(z_axis, -1)
    return out


class ApparentSource(nn.Module):
    """PyTorch module for predicting Q1, Q2 and maybe Q3"""

    def __init__(self,
                 mean,
                 scale,
                 inputs=(('LHF', 1), ('SHF', 1), ('SOLIN', 1), ('QT', 34),
                         ('SLI', 34), ('FQT', 34), ('FSLI', 34)),
                 outputs=(('SLI', 34), ('QT', 34))):

        "docstring"
        super(ApparentSource, self).__init__()

        self.inputs = VariableList.from_tuples(inputs)
        self.outputs = VariableList.from_tuples(outputs)

        self.mean = mean
        self.scale = scale
        self.scaler = scaler(scale, mean)
        n = 256

        self.model = nn.Sequential(
            LinearDictIn([(x.name, x.num) for x in self.inputs], n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            LinearDictOut(n, [(x.name, x.num) for x in self.outputs]))

    def forward(self, x):
        """Estimated source terms and diagnostics

        All inputs have shape (*, z, y, x) or (*, y, x)

        """
        reshaped_inputs = _height_to_last_dim(x, input_specs=self.inputs)
        scaled = self.scaler(reshaped_inputs)
        out = self.model(scaled)
        return _height_to_original_dim(out, self.outputs)

    def call_with_xr(self, ds, **kwargs):
        """Call the neural network with xarray inputs"""
        tensordict = _dataset_to_torch_dict(ds)
        with torch.no_grad():
            output = self(tensordict, **kwargs)
        return _torch_dict_to_dataset(output, ds.coords)


class ForcedStepper(nn.Module, SaverMixin):
    """Single Column Model Stepper

    Integrates the single column equations with the trapezoid rule:

        x_n+1 = x_n + f(x_n) dt + (g(t_n) + g(t_n+1)) dt / 2

    """

    def __init__(self,
                 mean,
                 scale,
                 time_step,
                 auxiliary=(),
                 prognostic=(),
                 diagnostic=(),
                 forcing=()):
        super(ForcedStepper, self).__init__()

        # store arguments for saver mixin
        self.kwargs = locals()
        self.kwargs.pop('self')
        self.kwargs.pop('__class__')

        self.time_step = torch.tensor(float(time_step), requires_grad=False)
        self.auxiliary = auxiliary
        self.prognostic = prognostic
        self.diagnostic = diagnostic
        self.forcing = forcing

        self.rhs = ApparentSource(
            mean,
            scale,
            inputs=auxiliary + prognostic,
            outputs=prognostic + diagnostic)

    def _get_sequence_length(self, x):
        return x[first(self.prognostic)[0]].size(0) - 1

    def _get_auxiliary_for_step(self, x, t):
        return {key: x[key][t] for key, _ in self.auxiliary}

    def _get_prognostic_for_step(self, x, t):
        return {key: x[key][t] for key, _ in self.prognostic}

    def _get_forcing_for_step(self, x, t):
        """Use trapezoid rule to get the forcing"""
        return {
            key: (x['F' + key][t] + x['F' + key][t + 1]) / 2
            for key in self.forcing
        }

    def _update_prognostics_and_diagnostics(self, prog, forcing, nn_output, dt=None):
        if dt is None:
            dt = self.time_step
        else:
            dt = torch.tensor(float(dt)).float()

        for key, num in self.prognostic:
            # apparent source should be units [key]/seconds
            apparent_src = nn_output.pop(key) / 86400
            prog[key] = prog[key] + apparent_src * dt
            # store neural network diagnostics for layer
            nn_output['F' + key + 'NN'] = apparent_src
            # add the large-scale forcing
            for key in forcing:
                prog[key] = prog[key] + self.time_step * forcing[key]

    def disable_forcing(self):
        self._forcing = self.forcing
        self.forcing = ()

    @property
    def heights(self):
        n = max(self.rhs.inputs.nums)
        return range(n)

    def forward(self, x, n=None, n_prog=None, dt=None):
        """Produce an n-step prediction for single column mode

        Parameters
        ----------
        x : dict
            a dict of torch tensors containing all the keys in the prognostic, diagnostic, and forcing attributes
        n : int
            the number of time steps to produce a prediction for
        n_prog : int
            the number of burn-in-steps. The prediction only begins after n_prog steps.
        """
        x = x.copy()


        if n is None:
            n = self._get_sequence_length(x)

        if n_prog is None:
            n_prog = n

        predictions = []
        for t in range(n):
            if t < n_prog:
                prog = self._get_prognostic_for_step(x, t)
            aux = self._get_auxiliary_for_step(x, t)
            forcing = self._get_forcing_for_step(x, t)
            # call neural network
            nn_output = self.rhs(merge(aux, prog))
            self._update_prognostics_and_diagnostics(prog, forcing, nn_output, dt=dt)
            predictions.append(merge(nn_output, prog))

        return merge_with(torch.stack, predictions)

    def __repr__(self):
        return 'ForcedStepper'


def model_factory():
    return ForcedStepper
