from typing import Mapping

import numpy as np
import torch


# Basic utilities and calculus
def xarray2numpy(x):
    return np.asarray(x)


def torch2numpy(x):
    if isinstance(x, Mapping):
        return {key: torch2numpy(val) for key, val in x.items()}
    else:
        return x.detach().numpy()


def numpy2torch(x):
    if isinstance(x, Mapping):
        return {key: numpy2torch(val) for key, val in x.items()}
    else:
        return torch.tensor(x, requires_grad=True).float()
