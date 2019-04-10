from uwnet.stochastic_parameterization.residual_stochastic_state_model import (
    StochasticStateModel,
)
from uwnet.thermo import integrate_q2
from uwnet.stochastic_parameterization.utils import get_dataset
import xarray as xr
from matplotlib import pyplot as plt

dir_ = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/'
base_model_location = dir_ + 'full_model/1.pkl'
ds_location = dir_ + 'training.nc'

base_model = StochasticStateModel(
    return_stochastic_state=False,
    binning_quantiles=(1,),
    base_model_location=base_model_location,
    ds_location=ds_location,
    verbose=False)
base_model.train()

stochastic_model = StochasticStateModel(
    return_stochastic_state=False,
    base_model_location=base_model_location,
    ds_location=ds_location,
    binning_method='column_integrated_qt_residuals',
    verbose=False)
stochastic_model.train()

ds = get_dataset(ds_location=ds_location, set_eta=False)

input_ = ds.isel(time=[100])
true_qt_forcing = (
    ds.isel(time=101).QT -
    ds.isel(time=100).QT -
    (10800 * ds.isel(time=100).FQT)
) / .125
titles = []
precips = []
precips.append(-integrate_q2(true_qt_forcing, ds.layer_mass).expand_dims('time'))
titles.append('True Data')
precips.append(-integrate_q2(base_model.predict(input_).QT, ds.layer_mass))
titles.append('Base Model')
precips.append(-integrate_q2(stochastic_model.predict(input_).QT, ds.layer_mass))
titles.append('Stochastic Model')

for idx in range(1, 5):
    stochastic_model.simulate_eta(t_start=0, n_time_steps=100)
    precips.append(-integrate_q2(stochastic_model.predict(input_).QT, ds.layer_mass))
    titles.append(f'Stochastic State Model, Random Simulation #{idx}')
precips = xr.concat(precips, dim=titles).rename({'concat_dim': 'model'})
precips.isel(time=0).plot(col='concat_dim', col_wrap=3, aspect=2)
