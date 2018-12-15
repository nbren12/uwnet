import torch
from torch import nn

import uwnet.modules as um
from uwnet.normalization import Scaler

from .xarray_interface import XRCallMixin
from .pre_post import get_pre_post


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


def get_model(dataset, vertical_grid_size):
    """Create an MLP with scaled inputs and outputs
    """
    pre_type = 'pca'
    args = get_pre_post(pre_type, dataset, vertical_grid_size)
    return InnerModel(*args).to(dtype=torch.float)
