import matplotlib.pyplot as plt
import torch
import xarray as xr

from src.data import training_data
from uwnet.thermo import compute_apparent_source
from uwnet.xarray_interface import call_with_xr

import common

plt.style.use("tableau-colorblind10")

model_path = "../../models/265/5.pkl"

# open model
model = torch.load(model_path)
model.eval()

# get data
ds = xr.open_dataset(training_data).isel(
    step=0, time=slice(10, 12), x=slice(0, 1))
q1 = compute_apparent_source(ds.SLI, 86400 * ds.FSLI)
q2 = compute_apparent_source(ds.QT, 86400 * ds.FQT)
pres = ds.p[0]

# compute nn output
predicted_srcs = call_with_xr(model, ds)
q1_pred = predicted_srcs['SLI']
q2_pred = predicted_srcs['QT']


# plot the results
def plot(args):
    fig, axs = plt.subplots(
        2, 2, constrained_layout=True, sharey=True, sharex=True)
    q1, q1_pred, q2, q2_pred = args
    labels = ['Q1 Truth', 'Q1 Prediction', 'Q2 Truth', 'Q2 Prediction']

    for ax, arg, label in zip(axs.flat, args, labels):
        arg = arg.isel(time=0, x=0)
        m = common.get_vmax(arg.values)
        im = ax.pcolormesh(arg.y, pres, arg, vmin=-m, vmax=m, cmap='RdBu_r')
        fig.colorbar(im, ax=ax)
        ax.set_ylim([pres.max(), pres.min()])
        ax.set_title(label)


plot([q1, q1_pred, q2, q2_pred])
