import torch
from collections import OrderedDict
from toolz import get, pipe, merge
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
                out[key] = out[key] * self.scale[key]
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


class MOE(nn.Module):

    def __init__(self, m, n, n_experts):
        "docstring"
        super(MOE, self).__init__()

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(m, n),
                )
            for _ in range(n_experts)
        ])

        self.decider = nn.Sequential(
            nn.Linear(m, 256),
            nn.ReLU(),
            nn.Linear(256, n_experts),
            nn.Softmax(dim=1)
        )

    def poll_experts(self, x):
        return [expert(x) for expert in self.experts]

    def forward(self, x):
        weights = self.decider(x)
        ans = 0
        for val, w in zip(self.poll_experts(x),
                          weights.split(1, dim=-1)):
            ans = ans + w * val
        return ans


class MLP(nn.Module, StackerScalerMixin, SaverMixin):
    input_fields = ['LHF', 'SHF', 'SOLIN', 'qt', 'sl', 'FQT', 'FSL']
    output = OrderedDict([
        ('sl', 34),
        ('qt', 34),
    ])

    def __init__(self, mean, scale):
        "docstring"
        super(MLP, self).__init__()


        nz = 34
        n2d = 3
        m = nz * 4 + n2d
        n = 2*nz

        # self.mod = MOE(m, 2*nz, n_experts40)
        self.mod = nn.Sequential(
            nn.Linear(m, 512),
            nn.ReLU(),
            nn.Linear(512, n)
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
    def aux(self):
        return set(self.input_fields) - set(self.output.keys())

    @property
    def args(self):
        return (self.mean, self.scale)

    def step(self, x, *args):
        stacked = pipe(x, self.scaler, self._stacked)
        out = self.mod(stacked)
        out = pipe(out, self._unstacked, self._unscale)
        out = {key: out[key] + x[key] for key in self.progs}
        out['qt'].clamp_(min=0.0)
        return out, None

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
            dictionary of output variables including prognostics and
            diagnostics

        """
        nt = x['LHF'].size(1)
        if n is None:
            n = nt
        output = []

        aux = {key: x[key] for key in self.aux}

        for t in range(0, nt):
            if t < n:
                progs = {key: x[key][:, t] for key in self.progs}

            inputs = merge(utils.select_time(aux, t), progs)
            out, _ = self.step(inputs)
            progs = {key: out[key] for key in self.progs}

            output.append(out)

        return utils.stack_dicts(output)
