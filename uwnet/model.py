import torch
from torch import nn

import uwnet.modules as um
from uwnet.normalization import Scaler

from .xarray_interface import XRCallMixin


class InnerModel(nn.Module, XRCallMixin):
    """Inner model which operates with height along the last dimension"""

    def __init__(self, mean, scale, inputs, outputs):
        "docstring"
        super(InnerModel, self).__init__()

        n = 256

        transpose = um.ValMap(um.RPartial(torch.transpose, -3, -1))
        self.scaler = Scaler(mean, scale)
        self.model = nn.Sequential(
            self.scaler,
            transpose,
            um.LinearDictIn([(name, num) for name, num in inputs], n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            um.LinearDictOut(n, [(name, num) for name, num in outputs]),
            transpose, )

    @property
    def scale(self):
        return self.scaler.scale

    def forward(self, x):
        q0 = self.scale['QT']
        q0 = q0.clamp(max=1)
        y = self.model(x)
        y['QT'] = y['QT'] * q0
        return y


def get_model(mean, scale, vertical_grid_size):
    """Create an MLP with scaled inputs and outputs
    """

    inputs = [('QT', vertical_grid_size), ('SLI', vertical_grid_size),
              ('SST', 1), ('SOLIN', 1)]

    outputs = (('QT', vertical_grid_size), ('SLI', vertical_grid_size))
    return InnerModel(mean, scale, inputs, outputs)
