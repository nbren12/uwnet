import torch
from torch import nn
import xarray as xr
import uwnet.modules as um
import warnings
from torch.serialization import SourceChangeWarning

from .xarray_interface import XRCallMixin

warnings.filterwarnings("ignore", category=SourceChangeWarning)


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

    def predict(self, x):
        outputs = []
        for time in x.time:
            row_for_time = x.sel(time=time)
            output_for_time = self.call_with_xr(row_for_time)
            outputs.append(output_for_time)
        return xr.concat(outputs, dim='time')


def get_model(pre, post, _config):
    """Create an MLP with scaled inputs and outputs
    """
    kind = _config['kind']

    if kind == 'inner_model':
        return InnerModel(pre, post).to(dtype=torch.float)
