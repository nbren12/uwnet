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


def plot_jacobian(jac):
    fig, axs = plt.subplots(2, 2, constrained_layout=True)

    for i, outkey in enumerate(jac.keys()):
        for j, inkey in enumerate(jac[outkey].keys()):
            ax = axs[i, j]
            val = jac[inkey][outkey]
            val = val.detach().numpy()
            im = ax.pcolormesh(val)
            ax.set_title(f"d{outkey}/dt from {inkey}")
            ax.set_xlabel("Pressure input variable")
            ax.set_ylabel("Pressure level response")

            plt.colorbar(im, ax=ax)


def main(model_path="../../models/265/5.pkl", y_index=32):

    # open model
    model = torch.load(model_path)
    model.eval()

    # get data
    ds = xr.open_dataset(ngaqua_climate_path)
    ds = ds.expand_dims('x', axis=-1)

    location = ds.isel(y=slice(y_index, y_index + 1))
    M = saliency_map_one_location(model, location)

    plot_jacobian(M)


# if __name__ == '__main__':
#     main()
