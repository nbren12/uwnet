import common
from uwnet.spectra import *

model_path = "../../nn/NNAll/20.pkl"

panel_specs = [
    ('a)', None),
    ('b)', {'q': 15}),
    ('c)', {'s': 19}),
    ('d)', {'q': 15, 's': 19}),
]

fig, axs = plt.subplots(
    2, 2, figsize=(common.width, common.width), sharex=True, sharey=True,
    constrained_layout=True)

for ax, (title, lrf_lid) in zip(axs.flat, panel_specs):

    print(f"Plotting {title}")
    coupler, mean = get_coupler(model_path, lrf_lid=lrf_lid)
    eig = compute_spectrum(coupler)
    im = scatter_spectra(eig, ax=ax, cbar=False)
    ax.set_title(title, loc="left")

fig.colorbar(im, ax=axs.tolist(), shrink=.5)



fig.savefig("spectra_input_vertical_levels.pdf")
