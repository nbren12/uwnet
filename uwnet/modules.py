import math

import numpy as np
import torch
from torch import nn

from .tensordict import TensorDict


class LinearDictIn(nn.Module):
    """A Dense Linear Layer Transform for dicts of tensors
    """

    def __init__(self, inputs, n_out, cls=nn.Linear):
        super(LinearDictIn, self).__init__()
        self.models = {}
        for field, num in inputs:
            self.models[field] = cls(num, n_out)
            # register submodule
            self.add_module(field, self.models[field])

        self.bias = nn.Parameter(torch.Tensor(n_out))
        self.reset_parameters()

    def forward(self, input):
        return sum(self.models[key](input[key])
                   for key in self.models) + self.bias

    def reset_parameters(self):
        n = sum(lin.weight.size(1) for lin in self.models.values())
        stdv = 1. / math.sqrt(n)
        for key in self.models:
            self.models[key].weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


class LinearDictOut(nn.Module):
    """A Dense Linear Layer Transform for dicts of tensors
    """

    def __init__(self, n_in, outputs, cls=nn.Linear):
        super(LinearDictOut, self).__init__()
        self.models = {}
        for field, num in outputs:
            self.models[field] = cls(n_in, num)
            # register submodule
            self.add_module(field, self.models[field])

    def forward(self, input):
        return TensorDict(
            {key: self.models[key](input)
             for key in self.models})


def get_affine_transforms(func, n):
    """Get weights matrix and bias of an affine function

    func(I) = A + b
    func(0) = b

    """
    identity = np.eye(n)
    z = np.zeros((1, n))

    b = func(z)
    A = func(identity) - b
    return A, b


class LinearFixed(nn.Module):
    """A linear tranform with fixed weights and bias

    Useful for representing scikit-learn affine transform functions
    """

    def __init__(self, *args):
        super(LinearFixed, self).__init__()

        for arg in args:
            arg.requires_grad = False

        weight, bias = args
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def forward(self, x: torch.tensor):
        return torch.matmul(x, self.weight) + self.bias

    @property
    def in_features(self):
        return self.weight.size(0)

    @property
    def out_features(self):
        return self.weight.size(1)

    @classmethod
    def from_affine(cls, func, n):
        """Initialize from an arbitrary python function

        Parameters
        ----------
        func
            function has input dimension n and output dimension m
        n : int
            input dimensions
        """
        args = get_affine_transforms(func, n)
        args = [torch.tensor(arg, requires_grad=True) for arg in args]
        return cls(*args)

    def __repr__(self):
        return f"LinearFixed({self.in_features}, {self.out_features})"


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


class _Partial(nn.Module):
    after = True

    def __init__(self, fun, *args, **kwargs):
        "docstring"
        super(_Partial, self).__init__()
        self.fun = fun
        self.args = args
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        if self.after:
            fargs = args + self.args
        else:
            fargs = self.args + args

        fkwargs = self.kwargs.copy()
        fkwargs.update(kwargs)
        return self.fun(*fargs, **fkwargs)


class Partial(_Partial):
    """Module similar to functools.partial

    This allows a function to be easily wrapped as a torch module. Which makes
    it compatible with nn.Sequential and easy to save/load.
    """
    after = False


class RPartial(_Partial):
    """Same as Partial but stored args are passed after the first argument"""
    after = True


class ValMap(nn.Module):
    def __init__(self, fun):
        "docstring"
        super(ValMap, self).__init__()
        self.fun = fun

    def forward(self, x):
        from .tensordict import TensorDict
        out = {}
        for key, val in x.items():
            out[key] = self.fun(val)
        return TensorDict(out)


def mapbykey(funcs, d):
    return {key: funcs[key](d[key]) for key in funcs}


class MapByKey(nn.Module):
    """Apply modules to inputs by key

    This code explains the how this module works::

        {key: funcs[key](d[key]) for key in funcs}

    Examples
    --------
    >>> a = torch.ones(1)
    >>> mapbykey({'a': lambda x: x}, {'a': a})
    """

    def __init__(self, funcs):
        super(MapByKey, self).__init__()
        self.funcs = nn.ModuleDict(funcs)

    def __getitem__(self, key):
        return self.funcs[key]

    @property
    def outputs(self):
        return [(key, self.funcs[key].out_features) for key in self.funcs]

    @property
    def inputs(self):
        return [(key, self.funcs[key].in_features) for key in self.funcs]

    def forward(self, d):
        return TensorDict(mapbykey(self.funcs, d))


class ConcatenatedWithIndex(nn.Module):
    """Module for concatenating the output of another module in a reproducable
    order"""
    def __init__(self, model, dim=-1):
        "docstring"
        super(ConcatenatedWithIndex, self).__init__()
        self.model = model
        self.dim = dim

        keys = []
        index = []
        for key, n in self.model.outputs:
            keys.append(key)
            for k in range(n):
                index.append(k)

        self.keys = keys
        self.index = torch.tensor(index)

    def forward(self, x):
        output = []
        y = self.model(x)
        for key in self.keys:
            output.append(y[key])
        return torch.concatenate(output, self.dim)
