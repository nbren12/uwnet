import torch
from collections import OrderedDict
from toolz import get, pipe, merge_with, merge
from torch import nn

from lib.torch.normalization import scaler
from . import utils


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
            if key in self.scale:
                out[key] = out[key] * self.scale[key] + self.mean[key]
        return out


class SaverMixin(object):
    """Mixin for output and initializing models from dictionaries of the state and
    arguments"""

    def to_dict(self):
        return {'args': self.args, 'state': self.state_dict()}

    @classmethod
    def from_dict(cls, d):
        mod = cls(*d['args'])
        mod.load_state_dict(d['state'])
        return mod


class SimpleLSTM(nn.Module, StackerScalerMixin, SaverMixin):
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

    @property
    def args(self):
        return (self.mean, self.scale)

    def forward(self, x, hidden):

        stacked = pipe(x, self.scaler, self._stacked)
        h, c = self.lstm(stacked, hidden)
        out = self.lin(h)

        # unstack
        out = pipe(out, self._unstacked, self._unscale)
        return out, (h, c)


class MLP(nn.Module, StackerScalerMixin, SaverMixin):
    input_fields = ['LHF', 'SHF', 'SOLIN', 'qt', 'sl', 'FQT', 'FSL']
    output = OrderedDict([
            ('sl', 34),
            ('qt', 34),
        ])

    def __init__(self, mean, scale):
        "docstring"
        super(MLP, self).__init__()

        self.hidden_dim = 256

        nz = 34
        n2d = 3
        m = nz * 4 + n2d

        self.mod = nn.Sequential(
            nn.Linear(m, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 2 * nz)
        )

        self.mean = mean
        self.scale = scale
        self.scaler = scaler(scale, mean)

    def init_hidden(self, *args, **kwargs):
        return None

    @property
    def progs(self):
        return set(self.input_fields) & set(self.output.keys())

    @property
    def diags(self):
        return set(self.output.keys()) - set(self.input_fields)

    @property
    def args(self):
        return (self.mean, self.scale)

    def step(self, x, *args):
        stacked = pipe(x, self.scaler, self._stacked)
        out = self.mod(stacked)
        out = pipe(out, self._unstacked, self._unscale)
        return out, None

    def forward(self, x, y):
        """
        Parameters
        ----------
        x
            prognostic variables. Initial condition only
        y
            auxiliary variables. Multiple time points

        Returns
        -------
        outputs
            dictionary of output variables including prognostics and diagnostics

        """
        nt  = y['LHF'].size(1)
        output = []
        for t in range(nt):
            inputs = merge(utils.select_time(y, t), x)
            out, _ = self.step(inputs)
            x = {key: out[key] for key in self.progs}

            output.append(out)

        return utils.stack_dicts(output)
