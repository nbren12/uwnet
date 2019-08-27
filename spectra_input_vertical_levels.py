import common
from uwnet.spectra import *
import pandas as pd

model_path = "../../nn/NNAll/20.pkl"

panel_specs = [
    ('a)', None),
    ('b)', {'q': 15}),
    ('c)', {'s': 19}),
    ('d)', {'q': 15, 's': 19}),
]


def get_data() -> xr.Dataset:
    titles = []
    eigs = []
    for title, lrf_lid in panel_specs:
        coupler, mean = get_coupler(model_path, lrf_lid=lrf_lid)
        eig = compute_spectrum(coupler)
        eigs.append(eig)
        titles.append(title)

    return xr.concat(eigs, dim=pd.Index(titles, name='title'))


def plot(eigs, xlim=None, ylim=None, **kwargs):

    fig, axs = plt.subplots(
        2, 2, figsize=(common.width, common.width), sharex=True, sharey=True,
        constrained_layout=True)

    for ax, (title, lrf_lid) in zip(axs.flat, panel_specs):
        print(f"Plotting {title}")
        eig = eigs.sel(title=title)
        im = scatter_spectra(eig, ax=ax, cbar=False, **kwargs)
        ax.set_title(title, loc="left")
    fig.colorbar(im, ax=axs.tolist(), shrink=.5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


data = get_data()
plot(data)
plt.savefig("spectra_input_vertical_levels.pdf")

plot(data, xlim=[-10, 10], ylim=[-1, None], symlogy=False)
plt.savefig("spectra_input_vertical_levels_zoom.pdf")
