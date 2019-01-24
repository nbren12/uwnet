import numpy as np
import xarray as xr
from matplotlib import colors
import matplotlib.pyplot as plt

import torch
from src.data import open_data, runs
from uwnet.thermo import lhf_to_evap, integrate_q2
import common


# Example of making your own norm.  Also see matplotlib.colors.
# From Joe Kington: This one gives two different linear ramps:
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def get_data():
    ds = open_data('training')
    time = ds.time[0] + 2

    # compute expected_precip
    model = torch.load('../../data/runs/model268-epoch5.debiased/model.pkl')
    data_at_time = ds.sel(time=time).load()
    neural_net_srcs = model.xmodel(data_at_time)
    semi_prognostic = -integrate_q2(neural_net_srcs['QT'], ds.layer_mass)

    # net precip from model
    net_precip = runs['debias'].data_2d.NPNN

    evap = lhf_to_evap(ds.LHF)
    net_precip_truth = ds.Prec - evap

    return xr.Dataset({
        'NG-Aqua': net_precip_truth.interp(time=time),
        'SemiProg': semi_prognostic,
        'NN': net_precip.interp(time=time),
    }).compute()


def plot(data):
    norm = MidpointNormalize(midpoint=0, vmin=-20, vmax=50)
    data = data.assign_coords(x=data.x / 1e6, y=data.y / 1e6)
    plotme = data.to_array(dim='Run')

    kwargs = dict(cmap='RdBu_r', norm=norm, rasterized=True)

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(common.textwidth, common.textwidth / 3.5),
        sharex=True,
        sharey=True,
        constrained_layout=True)

    for k, run in enumerate(plotme.Run.values):
        val = plotme.sel(Run=run)
        im = axs[k].pcolormesh(val.x, val.y, val, **kwargs)
        # axs[k].set_aspect(1.0)

    fig.colorbar(im, ax=axs.tolist(), pad=-.02, aspect=60)

    common.label_outer_axes(axs[np.newaxis], "x (1000 km)", "y (1000 km)")

    axs[0].set_title("a) NG-Aqua", loc='left')
    axs[1].set_title("b) NN Diagnosis", loc='left')
    axs[2].set_title("c) NN+SAM Simulation", loc='left')


plot(get_data())
