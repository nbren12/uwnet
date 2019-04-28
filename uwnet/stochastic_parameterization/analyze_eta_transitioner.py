from uwnet.stochastic_parameterization.stochastic_state_model import (
    StochasticStateModel,
)
from uwnet.thermo import integrate_q2, compute_apparent_source
from uwnet.stochastic_parameterization.utils import (
    get_dataset,
    default_eta_transitioner_predictors,
)
from matplotlib import pyplot as plt
import xarray as xr

dir_ = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/'
base_model_location = dir_ + 'full_model/1.pkl'
ds_location = dir_ + 'training.nc'

ds = get_dataset(ds_location=ds_location, set_eta=False)
input_ = ds.isel(time=[100])

base_model = StochasticStateModel(
    binning_quantiles=(1,),
    base_model_location=base_model_location,
    ds_location=ds_location,
    verbose=False)
base_model.train()


def plot_predicted_q2(base_model, stochastic_model):
    true_qt_forcing = compute_apparent_source(
        ds.QT, ds.FQT * 86400).isel(time=100)
    titles = []
    precips = []
    precips.append(
        -integrate_q2(true_qt_forcing, ds.layer_mass).expand_dims('time'))
    titles.append('True Data')
    precips.append(
        -integrate_q2(base_model.predict(input_).QT, ds.layer_mass))
    titles.append('Base Model')
    precips.append(
        -integrate_q2(stochastic_model.predict(
            input_, eta=stochastic_model.eta).QT, ds.layer_mass))
    titles.append('True Eta')

    for idx in range(1, 4):
        stochastic_model.simulate_eta(t_start=0, n_time_steps=100)
        precips.append(
            -integrate_q2(stochastic_model.predict(input_).QT, ds.layer_mass))
        titles.append(f'Eta Simulation #{idx}')
    precips = xr.concat(precips, dim=titles).rename({'concat_dim': 'model'})
    precips.isel(time=0).plot(col='model', col_wrap=3, aspect=2)
    plt.show()


if __name__ == '__main__':
    eta_transitoner_predictors_with_state = [
        'moistening_pred',
        'heating_pred'] + default_eta_transitioner_predictors
    stochastic_model = StochasticStateModel(
        base_model_location=base_model_location,
        ds_location=ds_location,
        binning_method='column_integrated_qt_residuals',
        time_idx_to_use_for_eta_initialization=0,
        eta_coarsening=2,
        t_start=100,
        t_stop=150,
        verbose=False,
        markov_process=True,
        include_output_in_transition_model=True,
        eta_predictors=eta_transitoner_predictors_with_state)
    stochastic_model.train()
    plot_predicted_q2(base_model, stochastic_model)
