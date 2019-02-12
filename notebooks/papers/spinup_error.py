import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data import open_data
from uwnet.columns import single_column_simulation
from uwnet.thermo import compute_apparent_source
import common


def predict_for_each_time(model, location, num_pred_steps=20, num_time=160):
    """Run a single column model initialized with each time step of a Dataset


    Returns
    -------
    output:  xr.Dataset
        Has QT, SLI, FQTNN, and FSLINN variables. The dimensions of which are
        (step, time, z, y, x).
    """
    prognostics = ['QT', 'SLI']

    initial_times = np.arange(num_time - num_pred_steps)

    outputs = {}

    for init_time in tqdm(initial_times):
        simulation = single_column_simulation(
            model,
            location,
            interval=(init_time, init_time + num_pred_steps),
            prognostics=prognostics)

        for k, t in enumerate(simulation.time.values):
            outputs[(t, k)] = simulation.isel(time=k).drop('time')

    # convert outputs into a dataset
    idx = pd.MultiIndex.from_tuples(outputs.keys(), names=['time', 'step'])

    return xr.concat(outputs.values(), dim=idx)\
             .unstack('concat_dim')


@common.cache
def get_data(**kwargs):
    # open model and training data
    model = torch.load("../../models/270/2.pkl")
    ds = open_data('training')

    # select x=0, y=32
    index = {'x': 0, 'y': 32}
    index = {key: slice(val, val + 1) for key, val in index.items()}
    location = ds.isel(**index)

    # run the single column models
    merged = predict_for_each_time(model, location)

    # compute Q2 for comparison
    location_subset = location.sel(time=merged.time)
    q2 = compute_apparent_source(location_subset.QT,
                                 location_subset.FQT * 86400)
    plotme = xr.concat(
        [q2.assign_coords(step='Truth'), merged.FQTNN], dim='step')

    return plotme


def plot(data):
    data.isel(step=[0, 1, 2, 3]).plot(col='step', x='time')


if __name__ == '__main__':
    plot(get_data(num_pred_steps=3))
    plt.savefig("spinup.png")
