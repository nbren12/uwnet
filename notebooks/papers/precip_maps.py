import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

import torch
from src.data import open_data, runs
from uwnet.thermo import lhf_to_evap, integrate_q2
import common


# Example of making your own norm.  Also see matplotlib.colors.
# From Joe Kington: This one gives two different linear ramps:

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

    # net precip micro
    data_2d = runs['micro'].data_2d
    micro = data_2d.Prec - lhf_to_evap(data_2d.LHF)

    evap = lhf_to_evap(ds.LHF)
    net_precip_truth = ds.Prec - evap

    return xr.Dataset({
        'NG-Aqua': net_precip_truth.interp(time=time),
        'SemiProg': semi_prognostic,
        'DEBIAS': net_precip.interp(time=time),
        'MICRO': micro.interp(time=time),
    }).compute()


def plot(data):
    norm = common.MidpointNormalize(midpoint=0, vmin=-20, vmax=50)
    data = data.assign_coords(x=data.x / 1e6, y=data.y / 1e6)
    plotme = data.to_array(dim='Run')

    kwargs = dict(cmap='RdBu_r', norm=norm, rasterized=True)

    fig = plt.figure(figsize=(common.textwidth, common.textwidth))
    grid = AxesGrid(
        fig,
        111,  # similar to subplot(144)
        nrows_ncols=(2, 2),
        aspect=True,
        axes_pad=(.2, .3),
        label_mode="L",
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="3%",
        cbar_pad="2%", )



    axs = np.array(list(grid)).reshape((2, 2))
    common.label_outer_axes(axs, "x (1000 km)", "y (1000 km)")

    for k, run in enumerate(plotme.Run.values):
        val = plotme.sel(Run=run)
        im = grid[k].pcolormesh(val.x, val.y, val, **kwargs)
        # axs[k].set_aspect(1.0)

    fig.colorbar(im, cax=grid.cbar_axes[0])

    grid[0].set_title("a) NG-Aqua", loc='left')
    grid[1].set_title("b) NN Semi-prognostic", loc='left')
    grid[2].set_title("c) DEBIAS Simulation", loc='left')
    grid[3].set_title("d) Micro Simulation", loc='left')


plot(get_data())
