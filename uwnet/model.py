from collections import OrderedDict

import attr
import torch
from toolz import first, get, merge, merge_with, pipe
from functools import partial
from torch import nn

import xarray as xr
from uwnet.modules import LinearDictIn, LinearDictOut
from uwnet.normalization import Scaler


def cat(seq):
    seq_with_nulldim = []
    sizes = []

    for x in seq:
        seq_with_nulldim.append(x)
        sizes.append(x.size(-1))

    return sizes, torch.cat(seq_with_nulldim, -1)


def _dataset_to_torch_dict(ds):
    return {
        key: torch.from_numpy(ds[key].values).float()
        for key in ds.data_vars
    }


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
        return pipe(inputs, _numpy_dict_to_torch_dict,
                    partial(self, **kwargs), _torch_dict_to_numpy_dict)


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
    return {
        spec.name: spec.move_height_to_last_dim(x[spec.name])
        for spec in input_specs
    }


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

    def __init__(self, model, inputs=(), outputs=()):
        "docstring"
        super(ApparentSource, self).__init__()
        self.model = model
        self.inputs = VariableList.from_tuples(inputs)
        self.outputs = VariableList.from_tuples(outputs)

    def forward(self, x):
        """Estimated source terms and diagnostics

        All inputs have shape (*, z, y, x) or (*, y, x)

        """
        reshaped_inputs = _height_to_last_dim(x, input_specs=self.inputs)
        out = self.model(reshaped_inputs)
        return _height_to_original_dim(out, self.outputs)

    def call_with_xr(self, ds, **kwargs):
        """Call the neural network with xarray inputs"""
        tensordict = _dataset_to_torch_dict(ds)
        with torch.no_grad():
            output = self(tensordict, **kwargs)
        return _torch_dict_to_dataset(output, ds.coords)


class InnerModel(nn.Module):
    """Inner model which operates with height along the last dimension"""

    def __init__(self, mean, scale, inputs, outputs):
        "docstring"
        super(InnerModel, self).__init__()

        n = 256

        self.scaler = Scaler(mean, scale)
        self.model = nn.Sequential(
            self.scaler,
            LinearDictIn([(name, num) for name, num in inputs], n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(), LinearDictOut(n, [(name, num) for name, num in outputs]))

    @property
    def scale(self):
        return self.scaler.scale

    def forward(self, x):
        q0 = self.scale['QT']
        q0 = q0.clamp(max=1)
        y = self.model(x)
        y['QT'] = y['QT'] * q0
        return y


def get_model(mean, scale, vertical_grid_size):
    """Create an MLP with scaled inputs and outputs
    """

    inputs = [('QT', vertical_grid_size), ('SLI', vertical_grid_size),
              ('SST', 1), ('SOLIN', 1)]

    outputs = (('QT', vertical_grid_size), ('SLI', vertical_grid_size))
    model = InnerModel(mean, scale, inputs, outputs)
    return ApparentSource(model, inputs, outputs)
