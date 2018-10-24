import matplotlib.pyplot as plt
import xarray as xr
import click
import torch
from uwnet.model import call_with_xr, model_factory


def column_integrate(data_array, mass):
    return (data_array * mass).sum('z')

def plot_water_imbalance(ds, mod, max_height, ax=None):

    if max_height:
        ds = ds.isel(z=slice(max_height))

    output = call_with_xr(mod, ds, drop_times=0)


    net_precip = -column_integrate(output.QT, ds.layer_mass).mean(['x', 'time']) / 1000
    cfqt = column_integrate(ds.FQT, ds.layer_mass).mean(['x', 'time']) * 86400 / 1000

    # plot data
    cfqt.plot(label='cFQT', ax=ax)
    net_precip.plot(label='Net Precipitation', ax=ax)
    ax.autoscale('y', tight=True)
    ax.legend()
    ax.set_ylabel('mm/day')


@click.command()
@click.argument('dataset')
@click.argument('model')
@click.option('-z', type=int, default=None, help='maximum height used in the network')
@click.option('-o', '--output-path', type=click.Path(), default=None, help='where to save figure')
def main(dataset, model, z, output_path=None):
    ds = xr.open_dataset(dataset).isel(time=slice(0, 120))
    mod = torch.load(model)
    fig, ax = plt.subplots()
    plot_water_imbalance(ds, mod, z, ax=ax)
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


if __name__ == '__main__':
    main()
