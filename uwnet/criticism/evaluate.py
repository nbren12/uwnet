import matplotlib.pyplot as plt
import xarray as xr
import click
import torch
from uwnet.xarray_interface import call_with_xr
import numpy as np
from uwnet.metrics import r2_score
import json

def column_integrate(data_array, mass):
    return (data_array * mass).sum('z')

def water_imbalance(ds, mod, max_height, ax=None):
    output = call_with_xr(mod, ds)

    net_precip = -column_integrate(output.QT, ds.layer_mass).mean(['x', 'time']) / 1000
    cfqt = column_integrate(ds.FQT, ds.layer_mass).mean(['x', 'time']) * 86400 / 1000

    bias = cfqt - net_precip
    ms_bias = float(np.sqrt((bias**2).mean()))


    return {"mean_squared_net_precip_bias": ms_bias}


@click.command()
@click.argument('dataset')
@click.argument('model')
@click.option('-z', type=int, default=None, help='maximum height used in the network')
@click.option('-o', '--output-path', type=click.Path(), default=None, help='where to save figure')
def main(dataset, model, z, output_path=None):
    model = torch.load(model)
    ds = xr.open_dataset(dataset).isel(time=slice(0, 120))
    imbalance = water_imbalance(ds, model, z)
    print(json.dumps(imbalance))



if __name__ == '__main__':
    main()
