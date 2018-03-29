import numpy as np
import pandas as pd
from pathlib import Path

import lib.cam as lc
from lib.interp import interp
import xarray as xr
from xnoah import swap_coord


def load_data(best_nn="model.VaryNHid-256/2",
              datadir="../../data"):

    # Truth

    root = Path(datadir)
    truth = xr.open_dataset(root / "processed/inputs.nc")
    force = xr.open_dataset(root / "processed/forcings.nc")
    truth = truth.assign(prec=force.Prec)
    p0 = truth.p

    # neural network scheme
    best_path = root / f"output/{best_nn}.columns.nc"
    nn_cols = xr.open_dataset(best_path)

    # load and interpolate CAM
    cam = xr.open_dataset(root / "output/scam.nc")
    cam = lc.to_sam_z(cam, p0, dim='p')
    cam = cam[['T', 'Q', 'prec']]

    # compute sl and qt
    cam['sl'] = cam['T'] + 9.81/1004 * p0.z
    cam['qt'] = cam['Q'] * 1000

    # swap coords
    truth = swap_coord(truth, {'z': 'p'})
    nn_cols = swap_coord(nn_cols.assign(p=p0), {'z': 'p'})
    cam = swap_coord(cam.assign(p=p0), {'z': 'p'})

    # combine data
    datasets = [truth, nn_cols, cam]
    variables = ['qt', 'prec', 'sl']
    time = np.unique(np.intersect1d(cam.time.values, truth.time.values))
    data = [ds[variables].sel(time=time) for ds in datasets]
    model_idx = pd.Index(['Truth', 'Neural Network', 'CAM'], name="model")


    ds = xr.concat(data, dim=model_idx)

    # change y coordinates
    y = (ds.y - np.median(ds.y))
    ds['y'] = y

    return ds


def compute_errors(metric, ds, **kwargs):

    mad = {}
    truth = ds.sel(model='Truth')

    for key in ['Neural Network', 'CAM']:
        if key in ds.model.values.tolist():
            mad[key] = metric(
                truth, ds.sel(model=key), **kwargs).sortby('time')

    mad['Mean'] = metric(truth, truth.mean(['x', 'time']),
                         **kwargs).sortby('time')
    mad['Persistence'] = metric(
        truth, truth.isel(time=0), **kwargs).sortby('time')

    return mad


def mean_squared_error(truth, pred, dims=('x', )):
    return ((truth - pred).fillna(0.0)**2).mean(dims)


def mean_absolute_dev(truth, pred, dims=('x', )):
    return (truth - pred).fillna(0.0).apply(np.abs).mean(dims)


def hide_xlabels(x):
    x.xaxis.set_ticklabels([])
    x.set_xlabel('')

def despine(ax, visible=('left', 'bottom')):
    for k in ax.spines:
        if k not in visible:
            ax.spines[k].set_color('none')
