import torch
from torch import nn

import uwnet.modules as um

from .xarray_interface import XRCallMixin


class InnerModel(nn.Module, XRCallMixin):
    """Inner model which operates with height along the last dimension"""

    def __init__(self, pre, post):
        "docstring"
        super(InnerModel, self).__init__()

        n = 256
        transpose = um.ValMap(um.RPartial(torch.transpose, -3, -1))
        self.input_names = [name for name, _ in pre.inputs]
        self.model = nn.Sequential(
            transpose,
            pre,
            um.LinearDictIn([(name, num) for name, num in pre.outputs], n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            um.LinearDictOut(n, [(name, num) for name, num in post.inputs]),
            post,
            transpose
        )

    def forward(self, x):
        return self.model(x[self.input_names])


class TransposedModel(nn.Module, XRCallMixin):
    """Inner model which operates with height along the last dimension"""

    def __init__(self, model):
        "docstring"
        super(InnerModel, self).__init__()

        transpose = um.ValMap(um.RPartial(torch.transpose, -3, -1))
        self.model = nn.Sequential(
            transpose,
            model,
            transpose
        )

    def forward(self, x):
        return self.model(x)


def get_causal_inner_model(pre, nodes=256, layers=3):
    pre = um.ConcatenatedWithIndex(pre)
    from IPython import embed; embed()


def get_model(pre, post, _config):
    """Create an MLP with scaled inputs and outputs
    """
    kind = _config['kind']

    if kind == 'inner_model':
        return InnerModel(*args).to(dtype=torch.float)
    elif kind == 'causal':
        inner = get_causal_inner_model(pre)
