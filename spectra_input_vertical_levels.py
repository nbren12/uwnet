from uwnet.wave.wave import ablate_upper_atmosphere, LinearResponseFunction, WaveCoupler, WaveEq
from uwnet.wave.spectra import compute_spectrum, scatter_spectra
import numpy as np

import common
from plots.common import WIDTH
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


LRF = "lrf/nn_NNAll_20.json"

PANEL_SPECS = [
    ('a) Full Atmosphere', {}),
    ('b) No Upper Humidity', {'q': 15}),
    ('c) No Upper Temperature', {'s': 19}),
    ('d) No Upper Humidity or \nTemperature', {'q': 15, 's': 19}),
]


def get_wave_coupler(path, lrf_lid):

    # open lrf
    with open(path) as f:
        lrf = LinearResponseFunction.load(f)

    lrf.panes = ablate_upper_atmosphere(lrf.panes, lrf_lid)

    # create a waveeq
    wave = WaveEq(lrf.base_state)

    # couple them
    return WaveCoupler(wave, lrf)


def get_data() -> xr.Dataset:
    titles = []
    eigs = []
    for title, lrf_lid in PANEL_SPECS:
        coupler = get_wave_coupler(LRF, lrf_lid=lrf_lid)
        eig = compute_spectrum(coupler)
        eigs.append(eig)
        titles.append(title)

    return xr.concat(eigs, dim=pd.Index(titles, name='title'))


def plot(eigs, xlim=None, ylim=None, **kwargs):

    fig, axs = plt.subplots(
        2, 2, figsize=(WIDTH, WIDTH), sharex=True, sharey=True)

    for ax, (title, lrf_lid) in zip(axs.flat, PANEL_SPECS):
        print(f"Plotting {title}")
        eig = eigs.sel(title=title)
        im = scatter_spectra(eig, ax=ax, cbar=False, **kwargs)
        ax.set_title(title, loc="left", fontsize=10)
    
    # remove xlabels for first row
    for ax in axs[0,:]:
        ax.set_xlabel('')

    # remove ylabels for last column
    for ax in axs[:,1]:
        ax.set_ylabel('')

    common.add_wavelength_colorbar(fig, im, ax=axs.tolist(), shrink=.5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


data = get_data()
plot(data)
plt.savefig("figs/spectra_input_vertical_levels.pdf")

plot(data, xlim=[-10, 10], ylim=[-1, None], symlogy=False)
plt.savefig("figs/spectra_input_vertical_levels_zoom.pdf")
