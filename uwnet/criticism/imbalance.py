import matplotlib.pyplot as plt
import xarray as xr
import click
from uwnet.model import call_with_xr, model_factory


def plot_water_imbalance(ds, mod):

    ds = ds.isel(z=mod.heights)

    # this call computes FQTNN and FSLINN for each ds snapshot
    output = call_with_xr(mod, ds)

    # compute water budget diagnostics
    PW_dot = ((output.FQTNN * ds.layer_mass).sum('z') +
              (ds.FQT * ds.layer_mass).sum('z')) / 1000 * 86400
    pw = (output.QT * ds.layer_mass).sum('z')
    pw_change = pw[-1] - pw[0]
    dt = (pw.time[-1] - pw.time[0])
    pw_change_mean = pw_change.mean('x') / 1000 / dt

    pw_change_mean.plot(label='PW Change')
    PW_dot.mean(['x', 'time']).plot(label='FQTNN + FQT')
    plt.autoscale()
    plt.ylabel("(mm/day)")
    plt.legend()

    plt.figure()
    cfqtnn = (output.FQTNN * ds.layer_mass).sum('z').mean(['x', 'time'])
    cfqt = (ds.FQT * ds.layer_mass).sum('z').mean(['x', 'time'])
    (ds.FQT * ds.layer_mass).sum('z').mean(['x', 'time']).plot()
    from IPython import embed; embed()


@click.command()
@click.argument('dataset')
@click.argument('model')
def main(dataset, model):
    ds = xr.open_dataset(dataset).isel(time=slice(0, 120))
    mod = model_factory().from_path(model)
    plot_water_imbalance(ds, mod)
    plt.show()


if __name__ == '__main__':
    main()
