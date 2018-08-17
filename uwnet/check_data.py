"""Script for validating input data"""
import sys
import xarray as xr
import numpy as np
import click


def _check_nan(x):
    sum_nan = int(np.isnan(x).sum())
    if sum_nan != 0:
        raise ValueError(
            f"NaNs detected in input. Total number is {sum_nan} of {x.size}.")


def check_necessary_variables_present(ds):
    variables_needed = {
        'SOLIN', 'QT', 'SLI', 'FSLI', 'FQT', 'U', 'V', 'W', 'SOLIN', 'SST',
        'RADTOA', 'RADSFC', 'Prec', 'LHF', 'SHF'
    }

    not_present = variables_needed - set(ds.data_vars)
    if len(not_present) > 0:
        raise ValueError("Not all needed variables are present. These "
                         f"variables are missing: {not_present}.")


def check_for_nans(ds):
    for key in ds:
        _check_nan(ds[key])


def check_units(ds):
    for key, expected in [('FQT', 'g/kg/s'), ('FSLI', 'K/s')]:
        actual = ds[key].units
        if actual != expected:
            raise ValueError(f"{key} units are {actual}")


def check_w_correlated_with_fqt(ds):

    corr_spatial = (ds.W * ds.FQT).mean(['x', 'y', 'time'])
    ans = (corr_spatial * ds.layer_mass).sum('z')
    if ans < 0:
        raise ValueError(
            "FQT and W are negatively correlated...check this data.")


def check_time_equally_spaced(ds):
    time = ds.time
    dt = np.diff(time)

    sig = dt.std() / dt.mean()
    print(f"Variance in dt is {sig}%")
    assert sig < 1e-6


def check_w_domain_mean_vanishes(ds):
    # vertical velocity should vanish
    mean = ds.W[0].mean(['x', 'y'])
    sig = ds.W[0].std(['x', 'y'])

    M = ds.layer_mass.sum()

    mean_ss = np.sqrt(float((ds.layer_mass * (mean**2)).sum() / M))
    sig_ss = np.sqrt(float((ds.layer_mass * (sig**2)).sum() / M))

    if (mean_ss / sig_ss) > 1e-4:
        raise ValueError(
            "The domain mean of W is larger than acceptable. The typical "
            f"magnitude is {mean_ss} while the standard deviation is {sig_ss}."
        )


def run_checks(ds):
    check_funs = [f for f in globals() if f.startswith('check')]
    failed = 0
    for func_name in check_funs:
        func = globals()[func_name]
        print(f"Running {func_name}...")
        try:
            func(ds)
        except Exception as exc:
            click.secho("Check BAD", fg="red")
            print(exc)
            failed += 1
        else:
            click.secho("Check OK", fg="blue")

    if failed > 0:
        click.secho(f"Failed {failed} of {len(check_funs)} checks!", fg="red")
        sys.exit(1)
    else:
        click.echo("All tests passed!")


data = sys.argv[1]
try:
    ds = xr.open_zarr(data)
except ValueError:
    ds = xr.open_dataset(data)

run_checks(ds)
