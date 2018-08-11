import click
import xarray as xr
from dask.diagnostics import ProgressBar
from uwnet import thermo


def compute_radiation(data):
    lw = data.pop('LWNS')
    sw = data.pop('SWNS')
    data['RADSFC'] = lw - sw

    lw = data.pop('LWNT')
    sw = data.pop('SWNT')
    data['RADTOA'] = lw - sw

    return data


def compute_forcings(data):
    pass


def load_data(paths):

    data = {}
    for info in paths:
        for field in info['fields']:
            try:
                ds = xr.open_dataset(info['path'], chunks={'time': 40})
            except ValueError:
                ds = xr.open_dataset(info['path'])

            data[field] = ds[field]

    # compute layer mass from stat file
    # remove presssur
    rho = data.pop('RHO')[0]
    rhodz = thermo.layer_mass(rho)

    data['layer_mass'] = rhodz

    # compute thermodynamic variables
    TABS = data.pop('TABS')
    QV = data.pop('QV')
    QN = data.get('QN', 0.0)
    QP = data.get('QP', 0.0)

    sl = thermo.liquid_water_temperature(TABS, QN, QP)
    qt = QV + QN

    data['sl'] = sl.assign_attrs(units='K')
    data['qt'] = qt.assign_attrs(units='g/kg')

    data = compute_radiation(data)
    compute_forcings(data)

    # rename staggered dimensions
    data['U'] = (data['U']
                 .rename({'xs': 'x', 'yc': 'y'}))

    data['V'] = (data['V']
                 .rename({'xc': 'x', 'ys': 'y'}))

    objects = [
        val.to_dataset(name=key).assign(x=sl.x, y=sl.y)
        for key, val in data.items()
    ]

    ds = xr.merge(objects, join='inner').sortby('time')
    return ds


@click.command()
@click.argument("config")
@click.argument("out")
def main(config, out):
    import yaml
    paths = yaml.load(open(config))['paths']
    with ProgressBar():
        load_data(paths)\
            .load()\
            .to_zarr(out)


if __name__ == '__main__':
    main()
