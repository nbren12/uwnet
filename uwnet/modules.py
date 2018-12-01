import torch
from torch import nn
import numpy as np
from .tensordict import TensorDict
import math


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
        return TensorDict({key: self.models[key](input) for key in
                           self.models})


def get_affine_transforms(func, n, m):
    """Get weights matrix and bias of an affine function"""
    identity = np.eye(n)
    z = np.zeros((n,))

    b = func(z)
    A = func(identity)
    return A, b


class LinearFixed(nn.Module):
    """A linear tranform with fixed weights and bias

    Useful for representing scikit-learn affine transform functions
    """

    def __init__(self, weight, bias):
        super(LinearFixed, self).__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def forward(self, x: torch.tensor):
        return torch.matmul(x, self.weight) + self.bias

    @classmethod
    def from_affine(cls, func, n, m):
        """Initialize from an arbitrary python function

        Parameters
        ----------
        func
            function has input dimension n and output dimension m
        n, m : int
            input and output dimensions
        """
        args = get_affine_transforms(func, n, m)
        args = [torch.tensor(arg, requires_grad=True) for arg in args]
        return cls(*args)


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
