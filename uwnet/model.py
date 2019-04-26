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


def get_model(pre, post, _config):
    """Create an MLP with scaled inputs and outputs
    """
    kind = _config['kind']

    if kind == 'inner_model':
        return InnerModel(pre, post).to(dtype=torch.float)
