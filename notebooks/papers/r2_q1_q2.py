import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

import common
from src.data import assign_apparent_sources, open_data
from uwnet.metrics import r2_score
from uwnet.thermo import integrate_q1, integrate_q2


@common.cache
def get_data():
    ds = open_data('training').pipe(assign_apparent_sources)

    model = torch.load('../../data/runs/model268-epoch5.debiased/model.pkl')
    outputs = model.xmodel(ds)
    for key in outputs:
        ds['F' + key + 'NN'] = outputs[key]

    output = xr.Dataset()

    output['net_moist_nn'] = integrate_q2(ds['FQTNN'], ds.layer_mass)
    output['net_heat_nn'] = integrate_q1(ds['FSLINN'], ds.layer_mass)
    output['net_moist'] = integrate_q2(ds['Q2'], ds.layer_mass)
    output['net_heat'] = integrate_q1(ds['Q1'], ds.layer_mass)
    output['Q1'] = ds['Q1']
    output['Q2'] = ds['Q2']
    output['Q1nn'] = ds['FSLINN']
    output['Q2nn'] = ds['FQTNN']

    return output


@common.cache
def get_r2s():
    ds = get_data()

    dims = ['x', 'time']

    data_vars = dict(
        q2=r2_score(ds.Q2, ds.Q2nn, dims),
        q1=r2_score(ds.Q1, ds.Q1nn, dims),
        heat=r2_score(ds.net_heat, ds.net_heat_nn, dims),
        moist=r2_score(ds.net_moist, ds.net_moist_nn, dims))

    return xr.Dataset(data_vars).pipe(common.to_pressure_dim)


def plot_r2_pane(ax, da, title):
    levels = np.arange(11) * .1
    da.plot.contourf(levels=levels, cmap='viridis', yincrease=False, ax=ax)
    ax.set_title(title)


def plot(ds):

    fig, (a, b, c) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    plot_r2_pane(a, ds.q1, 'Q1')
    plot_r2_pane(b, ds.q2, 'Q2')

    ds.heat.plot(label='Net heating', ax=c)
    ds.moist.plot(label='Net Moistening', ax=c)
    plt.ylim([0, 1.0])
    plt.legend()

    fig.subplots_adjust(hspace=.40)


if __name__ == '__main__':
    plot(get_r2s())
    plt.savefig("r2.png")
