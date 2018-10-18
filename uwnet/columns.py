import click
import xarray as xr
from uwnet.model import MLP


@click.command()
@click.argument('model')
@click.argument('data')
@click.argument('output_path')
def main(model, data, output_path):
    mod = MLP.from_path(model)
    ds = xr.open_dataset(data).isel(z=mod.z)
    mod.add_forcing = True
    output = mod.call_with_xr(ds, n=1)

    # compute water budget stuff
    w = ds.layer_mass
    output['layer_mass'] = w
    output['PW'] = (output.QT * w).sum('z') / 1000.0
    output['CFQTNN'] = (output.FQTNN * w).sum('z') / 1000.0
    output['CFQT'] = (ds.FQT * w).sum('z') / 1000.0

    # save to file
    output.to_netcdf(output_path)


if __name__ == '__main__':
    main()
