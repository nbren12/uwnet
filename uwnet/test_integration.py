"""Integration tests"""
import numpy as np
import pytest

import xarray as xr
from torch.utils.data import DataLoader
from uwnet.ml_models.nn.datasets_handler import XRTimeSeries
from uwnet.utils import batch_to_model_inputs


def _mock_dataset(dtype=np.float32):
    prognostic = [['b', 5]]
    auxiliary = [['a', 1]]
    diagnostic = [['c', 5]]
    forcing = ['b']

    inputs = prognostic + auxiliary
    outputs = prognostic + diagnostic

    loss_scale = {'b': 1.0, 'c': 1.0}

    t, z, y, x = (10, 5, 4, 3)

    dims_full = ['time', 'z', 'y', 'x']
    dims_2d = ['time', 'y', 'x']

    # make dataset
    data_vars = {}
    for name, num in inputs + outputs:
        if num == 1:
            data_vars[name] = (dims_2d, np.zeros((t, y, x), dtype=dtype))
        else:
            data_vars[name] = (dims_full, np.zeros(
                (t, num, y, x), dtype=dtype))

    coords = {
        dim: np.arange(shape, dtype=dtype)
        for dim, shape in zip(dims_full, (t, z, y, x))
    }

    data_vars['layer_mass'] = ('z', np.ones((z, ), dtype=dtype))
    data_vars['Fb'] = data_vars['b']

    ds = xr.Dataset(data_vars, coords=coords)
    ds.time.attrs['units'] = 's'

    return prognostic, auxiliary, diagnostic, forcing, loss_scale, ds


@pytest.mark.skip()
def test_datasets_model_integration():
    #TODO use the new model for this integration test
    batch_size = 4

    # load the data
    prognostic, auxiliary, diagnostic, forcing, loss_scale, ds = _mock_dataset(
    )
    train_data = XRTimeSeries(ds.load())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    constants = train_data.torch_constants()

    # initialize model
    lstm = ForcedStepper(
        train_data.mean,
        train_data.scale,
        train_data.timestep(),
        auxiliary=auxiliary,
        prognostic=prognostic,
        diagnostic=diagnostic,
        forcing=forcing)

    # get one batch from the data loader
    x = next(iter(train_loader))
    x = batch_to_model_inputs(x, auxiliary, prognostic, diagnostic, forcing,
                              constants)

    # run the outputs through the network
    y = lstm(x, n=1)

    # compute the loss
    criterion = MVLoss(loss_scale.keys(), constants['layer_mass'], loss_scale)

    print("Loss:", criterion(y, x).item())
