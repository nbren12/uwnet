"""Integration tests"""
import numpy as np
from toolz import merge

import xarray as xr
from torch.utils.data import DataLoader
from uwnet.datasets import XRTimeSeries
from uwnet.model import MLP
from uwnet.loss import MVLoss
from uwnet.utils import batch_to_model_inputs


def _mock_dataset(dtype=np.float32):
    inputs = [['a', 1], ['b', 5]]
    outputs = [['c', 5], ['b', 5]]
    forcings = [['b', 5]]

    loss_scale = {
        'b': 1.0,
        'c': 1.0
    }

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

    data_vars['layer_mass'] = ('z', np.ones((z,), dtype=dtype))
    data_vars['Fb'] = data_vars['b']

    ds = xr.Dataset(data_vars, coords=coords)
    ds.time.attrs['units'] = 's'

    return inputs, forcings, outputs, loss_scale, ds


def test_datasets_model_integration():
    batch_size = 4

    # load the data
    inputs, forcings, outputs, loss_scale, ds = _mock_dataset()
    train_data = XRTimeSeries(ds.load(), [['time'], ['x', 'y'], ['z']])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    constants = train_data.torch_constants()

    # initialize model
    lstm = MLP(
        train_data.mean,
        train_data.scale,
        time_step=float(train_data.timestep()),
        inputs=inputs,
        outputs=outputs,
        forcings=forcings
    )
    lstm.add_forcing = True

    # get one batch from the data loader
    x = next(iter(train_loader))
    x = batch_to_model_inputs(x, inputs, forcings, outputs, constants)

    # run the outputs through the network
    y = lstm(x, n=1)

    # compute the loss
    criterion = MVLoss(lstm.outputs.names, constants['layer_mass'], loss_scale)

    print("Loss:", criterion(y, x).item())
