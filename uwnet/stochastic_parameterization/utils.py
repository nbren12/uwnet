# Standard Library
from functools import lru_cache

# Thirdparty
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from PIL import Image
import torch
import xarray as xr

# uwnet
from uwnet.thermo import compute_apparent_source


# ---- data location ----
# model_dir = ''
model_dir = '/Users/stewart/projects/uwnet/uwnet/stochastic_parameterization/'  # noqa
model_location = model_dir + 'stochastic_model.pkl'
default_base_model_location = model_dir + 'full_model/1.pkl'
# default_ds_location = "training.nc"
default_ds_location = "uwnet/stochastic_parameterization/training.nc"
dataset_dt_seconds = 10800

# ---- binning method ----
# default_binning_method = 'precip'
default_binning_method = 'column_integrated_qt_residuals'
# default_binning_method = 'column_integrated_sli_residuals'

# ---- binning quantiles ----
default_binning_quantiles = (0.06, 0.15, 0.30, 0.70, 0.85, 0.94, 1)
# default_binning_quantiles = (.1, .3, .7, .9, 1)
# default_binning_quantiles = (1,)
# default_binning_quantiles = (.01, .05, .15, .35, .5, .65, .85, .95, .99, 1)

# ---- eta transitioner model ----
gbc = GradientBoostingClassifier(max_depth=500, verbose=2)
lr = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', max_iter=10000)
mlp = MLPClassifier(hidden_layer_sizes=(250,))
default_eta_transitioner_poly_degree = 3
default_eta_transitioner_model = lr
default_eta_transitioner_predictors = [
    'SST',
    'PW',
    'QT',
    'SLI',
    # 'FQT',
    # 'FSLI',
    # 'SHF',
    # 'LHF',
    'SOLIN',
    # 'RADSFC',
    # 'RADTOA',
    # 'U',
    # 'V'
    # 'FU',
    # 'FV'
]
residual_model_variables = [
    'heating_pred',
    'moistening_pred',
    'column_integrated_qt_residuals',
    'nn_moistening_residual',
    'nn_heating_residual'
]


def coarsen_array(array_, patch_width):
    if len(array_.shape) == 3:
        return np.stack([
            coarsen_array(array_2d, patch_width) for array_2d in array_])
    new_size = (
        int(array_.shape[1] / patch_width),
        int(array_.shape[0] / patch_width)
    )
    return np.array(Image.fromarray(array_.astype(float)).resize(
        new_size, Image.BOX))


def uncoarsen_2d_array(array_, new_size=(64, 128)):
    return np.array(Image.fromarray(array_.astype(float)).resize(
        (new_size[1], new_size[0])))


def uncoarsen_array(array_, new_size=(64, 128)):
    if len(array_.shape) == 4:
        return np.stack([
            uncoarsen_array(array_3d, new_size) for array_3d in array_])
    if len(array_.shape) == 3:
        return np.stack([
            uncoarsen_2d_array(array_2d, new_size) for array_2d in array_])
    return np.array(Image.fromarray(array_).resize((new_size[1], new_size[0])))


def blur(arr, sigma):
    if not {'x', 'y'} <= set(arr.dims):
        raise ValueError
    if sigma <= 1e-8:
        raise ValueError
    return gaussian_filter1d(
        gaussian_filter1d(
            arr, sigma, axis=arr.dims.index('x'), mode='wrap'),
        sigma, axis=arr.dims.index('y'), mode='nearest'
    )


def blur_dataset(data, sigma):
    for x in ['QT', 'SLI', 'PW']:
        data[x + '_blurred'] = data[x].copy()
        data[x + '_blurred'].values = blur(data[x], sigma)
    for x in ['moistening_pred', 'heating_pred']:
        data[x].values = blur(data[x], sigma)
    return data


def get_true_forcings(ds):
    qt_forcing = compute_apparent_source(ds.QT, ds.FQT * 86400)
    sli_forcing = compute_apparent_source(ds.SLI, ds.FSLI * 86400)
    return xr.Dataset({'QT': qt_forcing, 'SLI': sli_forcing})


def initialize_stochastic_model_features_for_dataset(dataset):
    dataset['column_integrated_qt_residuals'] = dataset['Prec'].copy()
    dataset['nn_moistening_residual'] = dataset['QT'].copy()
    dataset['nn_heating_residual'] = dataset['SLI'].copy()
    dataset['moistening_pred'] = dataset['QT'].copy()
    dataset['heating_pred'] = dataset['SLI'].copy()
    return dataset


def get_residual_model_variables(dataset, base_model_location):
    base_model = torch.load(base_model_location)
    dataset = initialize_stochastic_model_features_for_dataset(dataset)
    time_indices = list(range(len(dataset.time) - 1))
    true_forcings = get_true_forcings(dataset)

    for time_batch in np.array_split(time_indices, 10):
        pred = base_model.predict(dataset.isel(time=time_batch))
        true_qt = true_forcings.isel(time=time_batch).QT
        true_sli = true_forcings.isel(time=time_batch).SLI
        q2_preds = np.ma.average(
            pred['QT'],
            axis=1,
            weights=dataset.layer_mass.values
        )
        q2_true = np.ma.average(
            true_qt,
            axis=1,
            weights=dataset.layer_mass.values
        )
        dataset['column_integrated_qt_residuals'][
            time_batch, :, :] = q2_true - q2_preds
        dataset['moistening_pred'][time_batch, :, :, :] = pred['QT']
        dataset['heating_pred'][time_batch, :, :, :] = pred['SLI']
        dataset['nn_moistening_residual'][
            time_batch, :, :, :] = true_qt - pred['QT']
        dataset['nn_heating_residual'][
            time_batch, :, :, :] = true_sli - pred['SLI']
    return dataset


def get_bin_membership(
        dataset, binning_method, binning_quantiles, eta_coarsening):
    if binning_method == 'precip':
        binning_method_var = 'Prec'
    else:
        binning_method_var = binning_method
    binned_variable = dataset[binning_method_var]
    if eta_coarsening is not None:
        binned_variable = binned_variable.coarsen(
            {'x': eta_coarsening, 'y': eta_coarsening}).mean()
    bins = [
        binned_variable.quantile(quantile).values
        for quantile in binning_quantiles
    ]
    etas = np.digitize(binned_variable.values, bins, right=True)
    if eta_coarsening:
        etas = uncoarsen_array(etas)
    return etas


def insert_eta_bin_membership(
        dataset,
        binning_quantiles,
        base_model_location,
        binning_method,
        eta_coarsening):
    dataset = get_residual_model_variables(dataset, base_model_location)
    dataset = dataset.isel(time=range(len(dataset.time) - 1))
    dataset['eta'] = dataset['Prec'].copy()
    dataset['eta'].values = get_bin_membership(
        dataset, binning_method, binning_quantiles, eta_coarsening)
    dataset['eta_coarse'] = dataset['Prec'].copy()
    dataset['eta_coarse'].values = dataset['eta'].values
    # TODO: set eta_coarse differently than eta
    return dataset


def get_xarray_dataset_with_eta(
        data,
        binning_quantiles,
        base_model_location=None,
        eta_coarsening=None,
        t_start=0,
        t_stop=640,
        set_eta=True,
        blur_sigma=None,
        binning_method=default_binning_method):
    try:
        dataset = xr.open_zarr(data)
    except ValueError:
        dataset = xr.open_dataset(data)
    dataset = dataset.isel(time=range(t_start, t_stop))
    dataset['PW'] = (dataset.QT * dataset.layer_mass).sum('z') / 1000
    if set_eta:
        dataset = insert_eta_bin_membership(
            dataset,
            binning_quantiles,
            base_model_location,
            binning_method,
            eta_coarsening)
    if blur_sigma:
        dataset = blur_dataset(dataset, blur_sigma)
    try:
        return dataset.isel(step=0).drop('step').drop('p')
    except Exception:
        return dataset


@lru_cache()
def get_dataset(
        ds_location=default_ds_location,
        base_model_location=default_base_model_location,
        add_precipital_water=True,
        t_start=0,
        t_stop=640,
        set_eta=True,
        eta_coarsening=None,
        blur_sigma=None,
        binning_quantiles=default_binning_quantiles,
        binning_method=default_binning_method):
    ds = get_xarray_dataset_with_eta(
        ds_location,
        binning_quantiles,
        base_model_location=base_model_location,
        eta_coarsening=eta_coarsening,
        t_start=t_start,
        t_stop=t_stop,
        set_eta=set_eta,
        blur_sigma=blur_sigma,
        binning_method=binning_method
    )
    return ds


def count_transition_occurences(starting_indices, ending_indices):
    time_step_after_start = starting_indices.copy()
    time_step_after_start[:, 0] += 1
    set_a = set([tuple(coords) for coords in time_step_after_start])
    set_b = set([tuple(coords) for coords in ending_indices])
    return len(set_a.intersection(set_b))


def count_total_starting_occurences(dataset, indices_by_eta):
    max_time_idx = len(dataset.time) - 1
    return (indices_by_eta[:, 0] != max_time_idx).sum()
