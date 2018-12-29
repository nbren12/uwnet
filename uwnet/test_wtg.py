import torch
from .wtg import wtg_penalty
from .loss import weighted_mean_squared_error
from .timestepper import Batch
from .tensordict import TensorDict


def test_wtg():
    """This integration test ensures that the code runs

    TODO: this test depends on too much, I need to make it better
    """

    z = torch.arange(0, 10e3, 1e3)
    H = 5e3
    weights = torch.exp(-z / H)
    qt = torch.exp(-z / H) * 16.0
    sli = z * .004 + 290
    criterion = weighted_mean_squared_error(weights=weights, dim=-3)

    # shape is (batch, time, z, y, x)
    shape = (1, 1, -1, 1, 1)
    d = {'QT': qt.view(shape), 'SLI': sli.view(shape)}
    prognostics = ['QT', 'SLI']
    batch = Batch(d, prognostics)

    def model(x):
        return TensorDict({'QT': -x['QT'] / 5.0, 'SLI': -x['QT'] * 2})

    wtg_penalty(model, z, batch)
