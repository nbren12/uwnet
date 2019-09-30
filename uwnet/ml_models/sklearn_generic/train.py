import click
from dataclasses import dataclass, field
import joblib
import json
import logging
import os
import xarray as xr

# current has import error and does not recognize XarrayBatchLoader as importable
from .datasets_handler import XarrayLoaderSklearn
from uwnet.thermo import sec_in_day

# test with rf regressor first


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train_sklearn_model')




@dataclass
class ModelTrainingConfig:
    model_params: dict
    model_type: str

    data: str = "data/processed/reshaped/noBlur.zarr"
    existing_data_file: str = "data/processed/generic_sklearn_format/"
    predict_radiation: bool = True
    seed: int = 1234
    vertical_grid_size: int = 34
    num_samples: int = 20000
    num_samples_validation: int = 10000
    batch_size: int = 256
    num_batches_for_training: int = 5
    prognostics: list = field(default_factory=lambda: ['QT', 'SLI'])
    targets: list = field(default_factory=lambda: ['Q1', 'Q2'])

    def __post_init__(self):
        self.training_slices = slice(0, self.num_samples)
        self.validation_slices = slice(self.num_samples, self.num_samples + self.num_samples_validation)
        self.output_path = f"sklearn_models/{self.model_type}.pkl"

def get_model_training_config(model_config_path):
    with open(model_config_path, 'r') as f:
        config_dict = json.load(f)
    model_config = ModelTrainingConfig(**config_dict)
    return model_config


def get_dataset(config):
    logger.info(f"Opening xarray dataset {config.data}")
    try:
        dataset = xr.open_zarr(config.data)
    except ValueError:
        dataset = xr.open_dataset(config.data)

    if not config.predict_radiation:
        dataset['FSLI'] = dataset['FSLI'] + dataset['QRAD'] / sec_in_day

    try:
        return dataset.isel(step=0).drop('step').drop('p')
    except:
        return dataset


def get_data_loader(
        data: xr.Dataset,
        config,
        train
):
    if train:
        slices = config.training_slices
    else:
        slices = config.validation_slices
    data = data.isel(x=slices)

    # List needed variables
    variables = config.prognostics + ['SST', 'SOLIN', 'QRAD']
    for variable in config.prognostics:
        forcing_key = 'F' + variable
        variables.append(forcing_key)
    train_data = XarrayLoaderSklearn(
        dataset=data,
        batch_size=config.batch_size,
        variables=variables)
    return train_data



def get_regressor_model(model_config):
    if 'rf' in model_config.model_type or 'random_forest' in model_config.model_type:
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(**model_config.model_params)
        # temporarily set n_estimators to 0, will add estimators for each batch in iteration
        regressor.n_estimators = 0
    elif 'xgb' in model_config.model_type and 'tree' in model_config.model_type:
        # TODO: add other options?
        #from xgboost.sklearn_generic import XGBRegressor
        pass
    else:
        logger.error('model_type {self.model_type} provided in config file is not a valid model type.')

    return regressor

@click.command()
@click.option(
    '-cf',
    '--model-config-file',
    type=click.Path(),
    help='Location of config for model training.')
@click.option(
    '-op',
    '--model-output-path',
    type=click.Path(),
    default=None,
    help='Output the trained model to this path.')
def main(
        model_output_path,
        model_config_file,
):
    model_training_config = get_model_training_config(model_config_file)
    if not model_output_path:
        model_output_path = model_training_config.output_path
    dataset = get_dataset(model_training_config)
    train_loader = get_data_loader(dataset, model_training_config, train=True)
    regressor = get_regressor_model(model_training_config)

    for batch_num, training_batch in enumerate(train_loader):
        if batch_num >= model_training_config.num_batches_for_training:
            break
        train_X, train_y = training_batch
        if model_training_config.num_samples:
            train_X = train_X[:model_training_config.num_samples]
            train_y = train_y[:model_training_config.num_samples]
        logger.info(('training batch {0} / {1}'.format(batch_num, model_training_config.num_batches_for_training)))
        if 'rf' in model_training_config.model_type or 'random_forest' in model_training_config.model_type:
            regressor.n_estimators += model_training_config.model_params['n_estimators']

        regressor.fit(train_X, train_y)
    model_save_filename = os.path.join(
        model_output_path, f'{model_training_config.model_type}.pkl')
    joblib.dump(regressor, model_save_filename)


if __name__ == '__main__':
    main()
