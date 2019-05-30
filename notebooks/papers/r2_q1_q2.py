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
    model = common.get_model('NN-Lower')
    ds = open_data('training').pipe(assign_apparent_sources)
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
    im = ax.contourf(da.y, da.p, da,
                     levels=levels, cmap='viridis', extend='both')
    ax.invert_yaxis()
    # plt.colorbar(im, cax=cax)
    ax.set_title(title, loc='left')
    return im


def plot_line(ax, ds, title):
    ds.heat.plot(label='Net heating', ax=c)
    ds.moist.plot(label='Net Moistening', ax=c)
    plt.ylim([0, 1.0])
    plt.legend()

def plot(ds):

    ds = ds.assign_coords(y=ds.y/1e6)

    grid_spec_kws = dict(
        height_ratios=[1, 1, .7]
    )


    fig, (a, b, c) = plt.subplots(3, 1, constrained_layout=True, sharex=True,
                                  figsize=(4, 4),
                                  gridspec_kw=grid_spec_kws)

    plot_r2_pane(a, ds.q1, r'a) Apparent Heating ($Q_1$) $R^2$')
    im = plot_r2_pane(b, ds.q2, r'b) Apparent moistening ($Q_2$)  $R^2$')

    plt.colorbar(im, ax=[a, b], shrink=.6)

    c.plot(ds.y, ds.heat, label='Heating')
    c.plot(ds.y, ds.moist, label='Drying')
    c.yaxis.set_major_locator(plt.MaxNLocator(4))



    c.set_xlim(0, float(ds.y.max()))
    c.set_ylim(bottom=-.1)
    c.set_title('c) $R^2$ of vertical integral', loc='left')
    plt.legend(frameon=False, ncol=2)


    fig.set_constrained_layout_pads(w_pad=2./72., h_pad=2./72.,
                                    hspace=0.02, wspace=0.1)
    c.set_xlabel('y (1000 km)')
    for ax in (a, b):
        ax.set_ylabel('p (mb)')

    # fig.subplots_adjust(hspace=.40)


if __name__ == '__main__':
    data = get_r2s()
    plot(data)
    plt.savefig("r2.png")
    plt.savefig("r2.pdf")
