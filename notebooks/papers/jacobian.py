import matplotlib.pyplot as plt
import torch
import xarray as xr

from src.data import ngaqua_climate_path
from uwnet.thermo import compute_apparent_source
from uwnet.xarray_interface import dataset_to_torch_dict
from uwnet.jacobian import jacobian_from_model, jacobian


def saliency_map_one_location(model, ds):
    torch_data = dataset_to_torch_dict(ds)
    torch_data = torch_data[model.input_names]

    # compute saliency map
    return jacobian_from_model(model, torch_data)


def _get_vmax(val):
    a = val.min()
    b = val.max()

    return max([abs(a), abs(b)])


def plot_jacobian(args):
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    jac, p = args

    for i, outkey in enumerate(jac.keys()):
        for j, inkey in enumerate(jac[outkey].keys()):
            ax = axs[i, j]
            val = jac[inkey][outkey]
            val = val.detach().numpy()

            vmax = _get_vmax(val)

            im = ax.pcolormesh(
                p, p, val, cmap='RdBu_r', vmax=vmax, vmin=-vmax)
            ax.set_title(f"d{outkey}/dt from {inkey}")
            ax.set_xlabel("Pressure input variable")
            ax.set_ylabel("Pressure level response")
            ax.invert_yaxis()
            ax.invert_xaxis()

            plt.colorbar(im, ax=ax)


def main(model_path="../../models/265/5.pkl", y_index=32, **kwargs):

    # open model
    model = torch.load(model_path)
    model.eval()

    # get data
    ds = xr.open_dataset(ngaqua_climate_path)
    ds = ds.expand_dims('x', axis=-1)

    location = ds.isel(y=slice(y_index, y_index + 1))
    M = saliency_map_one_location(model, location)

    pres = ds.p.values.ravel()
    plot_jacobian([M, pres], **kwargs)


# if __name__ == '__main__':
#     main()
