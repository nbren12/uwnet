import click
import xarray as xr
from uwnet.model import MLP


@click.command()
@click.argument('model')
@click.argument('data')
@click.argument('output')
def main(model, data, output):
    mod = MLP.from_path(model)
    ds = xr.open_dataset(data).isel(z=mod.z)
    mod.call_with_xr(ds, n=1)\
        .to_netcdf(output)


if __name__ == '__main__':
    main()
