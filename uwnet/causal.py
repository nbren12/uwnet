import torch
from torch import nn
import math


def make_causal_mask(n, dependencies):
    dtype = torch.uint8

    cols = []
    for dependency in dependencies:
        col = torch.zeros(n, dtype=dtype)
        col[dependency] = 1
        cols.append(col)

    return torch.stack(cols, dim=1)


def make_causal_mask_from_ordinal(input, output, max=None, radius=1e-5):
    """Make a causal mask from an ordinal value

    Output j will have a value t^o(j) and input has value t^in(i). This output
    can depend only on inputs satisfying

    (t^in(i) <= t^o(j) AND t^in(i) < max) OR |t^in(i) - t^o(j)| < radius

    """
    mask = []
    input = input.float()
    output = output.float()
    if max is None:
        max = input[-1] + 10.0

    for val in output:
        col = (input <= val) * (input <= max)
        col[torch.abs(input - val) < radius] = 1
        mask.append(col)
    return torch.stack(mask, dim=1)


class CausalLinearBlock(nn.Module):
    """A linear block which forbids non-causal connections

    The user defines the notion of causality by ordering the inputs and outputs
    according to a list of real values. For example, one might assume that
    higher elevations cannot cause lower ones.

    Parameters
    ----------
    in_ordinal : torch.tensor
        has shape (*, n)
    out_ordinal : torch.tensor
        `(*, n)` This along with in_ordinal define the permitted connections
        within the weight matrix using :ref:`make_causal_mask_from_ordinal`.
        Has shape (*, m).
    activation
        The torch activation module to apply after. Defaults to ReLU.

    Returns
    -------
    out : torch.tensor
       Shape (*, m)
    """
    def __init__(self, in_ordinal, out_ordinal, activation=nn.ReLU()):
        super(CausalLinearBlock, self).__init__()

        self.activation = activation

        # setup mask
        self.register_buffer('in_ordinal', in_ordinal)
        self.register_buffer('out_ordinal', out_ordinal)
        mask = make_causal_mask_from_ordinal(in_ordinal, out_ordinal)
        self.register_buffer('mask', mask)

        # setup weights
        self._weight = nn.Parameter(
            torch.zeros(self.in_features, self.out_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features))

        # intialize weights
        self.reset_parameters()

    @property
    def in_features(self):
        return len(self.in_ordinal)

    @property
    def out_features(self):
        return len(self.out_ordinal)

    @property
    def weight(self):
        zero = torch.zeros(1)
        return self._weight.where(self.mask, zero)

    def forward(self, x):
        y = torch.matmul(x, self.weight) + self.bias
        if self.activation is None:
            return y
        else:
            return self.activation(y)

    def reset_parameters(self):
        n = self.out_features
        stdv = 1. / math.sqrt(n)
        self._weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
