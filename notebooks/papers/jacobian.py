import matplotlib.pyplot as plt
import torch
import xarray as xr

from src.data import ngaqua_climate_path
from uwnet.thermo import compute_apparent_source
from uwnet.xarray_interface import dataset_to_torch_dict
from uwnet.jacobian import jacobian_from_model, jacobian
from common import get_vmax


def saliency_map_one_location(model, ds):
    torch_data = dataset_to_torch_dict(ds)
    torch_data = torch_data[model.input_names]

    # compute saliency map
    return jacobian_from_model(model, torch_data)


def plot(args):
    fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(5, 4))
    jac, p = args

    k = 0

    abc = 'abcd'

    progs = list(jac)

    for i, outkey in enumerate(progs):
        for j, inkey in enumerate(progs):
            ax = axs[i, j]
            val = jac[outkey][inkey]
            val = val.detach().numpy()

            vmax = get_vmax(val)

            im = ax.pcolormesh(
                p, p, val, cmap='RdBu_r', vmax=vmax, vmin=-vmax)
            letter = abc[k]
            ax.set_title(f"{letter}) d{outkey}/dt from {inkey}", loc='left')
            ax.set_xlabel(f"Pressure ({inkey})")
            ax.set_ylabel(f"Pressure (d{outkey}/dt)")
            ax.invert_yaxis()
            ax.invert_xaxis()

            plt.colorbar(im, ax=ax, pad=-.04)
            k += 1

    return axs


def plot_with_dashes(args, qt_level=15, sli_level=18):
    """Chris wanted to see dashes where inputs are ignored"""
    axs = plot(args)

    p = args[1]

    # QT input
    for ax in axs[:, 0]:
        ax.axvline(p[qt_level], linestyle='--', c='k')

    # SLI input
    for ax in axs[:, 1]:
        ax.axvline(p[sli_level], linestyle='--', c='k')


def get_data(model_path="../../models/265/5.pkl", y_index=32,):
    # open model
    model = torch.load(model_path)
    model.eval()

    # get data
    ds = xr.open_dataset(ngaqua_climate_path)
    ds = ds.expand_dims('x', axis=-1)

    location = ds.isel(y=slice(y_index, y_index + 1))
    M = saliency_map_one_location(model, location)
    return M, ds.p.values.ravel()


def main(**kwargs):
    return plot_jacobian([M, pres], **kwargs)


# if __name__ == '__main__':
#     main()
