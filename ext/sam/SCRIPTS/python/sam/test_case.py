from .case import InitialConditionCase, Case, make_docker_cmd, pressure_correct
import numpy as np
import xarray as xr
import pytest


def test_case(tmpdir):
    z = np.r_[:25000:100j]
    case = Case(z, path=str(tmpdir))
    case.save()


def test_make_docker_cmd():
    truth = "docker run -w /pwd -v /a:/b -v /b:/c -e PATH=BS hello ls"
    cmd = make_docker_cmd(
        'hello',
        'ls',
        workdir='/pwd',
        volumes=[('/a', '/b'), ('/b', '/c')],
        env=[('PATH', 'BS')])

    assert ' '.join(cmd) == truth


def test_InitialConditionCase(tmpdir):
    z = xr.DataArray(np.r_[:25000:100j], dims=['z'], name='z')
    ds = z.to_dataset()
    case = InitialConditionCase(ic=ds, path=str(tmpdir))

    # test save method
    case.save()

    # test _z attribute
    assert case._z.tolist() == ds.z.values.tolist()


@pytest.fixture()
def init_condition_netcdf():
    path = str(pytest.config.rootdir.join("NGAqua/ic.nc"))
    return xr.open_dataset(path)


@pytest.mark.slow
def test_pressure_correct(tmpdir, init_condition_netcdf):
    """This is an integration test and takes about 5 seconds to run"""
    x = pressure_correct(
        init_condition_netcdf, path=str(tmpdir), sam_src=pytest.config.rootdir)
