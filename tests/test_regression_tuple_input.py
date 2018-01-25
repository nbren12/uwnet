"""This is a regression test that I will use for refactoring

It is just a temporary measure, and can be deleted when I am done changing the
format of the data input.

"""
import xarray as xr
from sklearn.externals import joblib
import torch
from lib.models.torch import train_multistep_objective, wrap
import pytest

@pytest.fixture()
def test_data():
    # open data
    def mysel(x):
        return x.isel(x=0, y=8).transpose('time', 'z')
    inputs="data/processed/inputs.nc"
    forcings="data/processed/forcings.nc"
    inputs = xr.open_dataset(inputs).pipe(mysel)
    forcings = xr.open_dataset(forcings).pipe(mysel)

    return inputs, forcings


@pytest.fixture()
def train_data():
    return joblib.load("data/ml/ngaqua/time_series_data.pkl")


def test_train_multistep_objective(train_data, test_data, regtest):


    inputs, forcings = test_data
    model = train_multistep_objective(train_data)
    wrapped = wrap(model)
    output = wrapped(inputs, forcings)

    print(output.isel(time=-1), file=regtest)



