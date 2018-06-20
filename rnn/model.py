import torch
from collections import OrderedDict
from toolz import get, pipe
from torch import nn

from lib.torch.normalization import scaler


def cat(seq):
    seq_with_nulldim = []
    sizes = []

    for x in seq:
        if x.dim() == 1:
            x = x[..., None]
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
        return cat(get(self.input_fields, x))[1]

    def _unstacked(self, out):
        out = out.split(tuple(self.output.values()), dim=-1)
        return dict(zip(self.output.keys(), out))

    def _unscale(self, out):
        # scale the outputs
        for key in self.output:
            out[key] = out[key] * self.scale[key] + self.mean[key]
        return out


class SimpleLSTM(nn.Module, StackerScalerMixin):
    def __init__(self, mean, scale):
        "docstring"
        super(SimpleLSTM, self).__init__()

        self.hidden_dim = 256
        self.input_fields = ['LHF', 'SHF', 'SOLIN', 'qt', 'sl', 'FQT', 'FSL']
        self.output = OrderedDict([
            ('sl', 34),
            ('qt', 34),
        ])

        nz = 34
        n2d = 3
        m = nz * 4 + n2d
        self.lstm = nn.LSTMCell(m, self.hidden_dim)
        self.lin = nn.Linear(self.hidden_dim, 2 * nz)
        self.mean = mean
        self.scale = scale
        self.scaler = scaler(scale, mean)

    def init_hidden(self, n, random=False):
        if random:
            return (torch.rand(n, self.hidden_dim) * 2 - 1,
                    torch.rand(n, self.hidden_dim) * 2 - 1)
        else:
            return (torch.zeros(n, self.hidden_dim), torch.zeros(
                n, self.hidden_dim))

    def forward(self, x, hidden):

        stacked = pipe(x, self.scaler, self._stacked)
        h, c = self.lstm(stacked, hidden)
        out = self.lin(h)

        # unstack
        out = pipe(out, self._unstacked, self._unscale)
        return out, (h, c)

    def to_dict(self):
        return {'args': (self.mean, self.scale), 'state': self.state_dict()}

    @classmethod
    def from_dict(cls, d):
        mod = cls(*d['args'])
        mod.load_state_dict(d['state'])
        return mod
