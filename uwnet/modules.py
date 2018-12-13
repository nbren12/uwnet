import torch
from torch import nn
import numpy as np
from .tensordict import TensorDict
import math


class LinearDictIn(nn.Module):
    """A Dense Linear Layer Transform for dicts of tensors
    """

    def __init__(self, inputs, n_out):
        super(LinearDictIn, self).__init__()
        self.models = {}
        for field, num in inputs:
            self.models[field] = nn.Linear(num, n_out, bias=False)
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

    def __init__(self, n_in, outputs):
        super(LinearDictOut, self).__init__()
        self.models = {}
        for field, num in outputs:
            self.models[field] = nn.Linear(n_in, num, bias=True)
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
