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
def get_data(model="../../models/277/1.pkl", **kwargs):
    # open model and training data
    model = torch.load("../../models/277/1.pkl")
    ds = open_data('training')

    # select x=0, y=32
    index = {'x': 0, 'y': 32}
    index = {key: slice(val, val + 1) for key, val in index.items()}
    location = ds.isel(**index)

    # run the single column models
    merged = predict_for_each_time(model, location, **kwargs)

    # compute Q2 for comparison
    location_subset = location.sel(time=merged.time)
    q2 = compute_apparent_source(location_subset.QT,
                                 location_subset.FQT * 86400)
    plotme = xr.concat(
        [q2.assign_coords(step='Truth'), merged.FQTNN], dim='step')

    return plotme


def get_title(step):
    if step == 0:
        return 'Truth'
    else:
        return f"Step = {step-1}"

def plot(data):
    p = open_data('pressure')

    fig, axs = plt.subplots(1, 4, sharex=True, sharey=True,
                            constrained_layout=True,
                            figsize=(common.textwidth, 2))
    axs.shape = (1, -1)
    abcd = 'abcd'

    m = common.get_vmax(data)
    kw = dict(levels=common.diverging_levels(25, 5), cmap='RdBu_r')
    for k in range(4):
        v = data.isel(step=k).squeeze()
        # import pdb; pdb.set_trace()
        im = axs[0,k].contourf(
            v.time, p, v.T, **kw)

        axs[0,k].set_title(f"{abcd[k]}) {get_title(k)}", loc='left')
        # v.plot(col='step', x='time')

    axs[0,0].invert_yaxis()
    plt.colorbar(im, ax=axs, orientation='horizontal',
                 shrink=.3, aspect=2)

    axs[0,0].yaxis.set_major_locator(plt.MaxNLocator(4))

    common.label_outer_axes(axs, "day", "p (mb)")




if __name__ == '__main__':
    data = get_data(model="../models/277/5.pkl", num_pred_steps=3, num_time=60)
    plot(data)
    plt.savefig("spinup_error.pdf")
