"""Train neural network parametrizations using the single or
multiple step loss function

This code uses a tool called Sacred_ to log important details about each
execution of this script to a database, and to specify hyper parameters of the
scheme (e.g. learning rate).

Examples
--------

This script can be executed with the following command::

   python -m uwnet.train with data=<path to dataset>

To see a list of all the available configuration options run::

   python -m uwnet.train print_config

.. _Sacred: https://github.com/IDSIA/sacred

"""
import logging
import os
from contextlib import contextmanager
from functools import partial
from os.path import join

import xarray as xr
from sacred import Experiment
from toolz import curry

import torch
from ignite.engine import Engine, Events
from torch.utils.data import DataLoader
from uwnet.datasets import XRTimeSeries, get_timestep
from uwnet.loss import get_input_output, get_step, weighted_mean_squared_error
from uwnet.model import get_model
from uwnet.pre_post import get_pre_post
from uwnet.training_plots import TrainingPlotManager

ex = Experiment("Q1", interactive=True)

XRTimeSeries = ex.capture(XRTimeSeries)
TrainingPlotManager = ex.capture(TrainingPlotManager, prefix='plots')
get_model = ex.capture(get_model, prefix='model')
get_pre_post = ex.capture(get_pre_post, prefix='prepost')


@ex.config
def my_config():
    """Default configurations managed by sacred"""
    data = "data/processed/training.nc"
    lr = .001
    epochs = 2
    model_dir = 'models'
    skip = 5
    time_length = None
    batch_size = 256
    vertical_grid_size = 34
    loss_scale = {
        'LHF': 150,
        'SHF': 10,
        'RADTOA': 600.0,
        'RADSFC': 600.0,
        'U': 5.0,
        'V': 5.0,
        'Prec': 8,
        'QP': 0.05,
        'QN': 0.05,
        'QT': 1.0,
        'SLI': 2.5
    }

    # y indices to use for training
    training_slices = dict(
        y=(None, None),
        x=(10, None),
        time=(None, None), )

    validation_slices = dict(
        y=(None, None),
        x=(0, 10),
        time=(None, None), )

    output_dir = None

    prognostics = ['QT', 'SLI']
    prepost = dict(kind='pca', path='models/prepost.pkl')

    model = dict(kind='inner_model')

    plots = dict(interval=1, single_column_locations=[(32, 0)])
    step = dict(
        name='instability',
        kwargs={'alpha': 1.0}
    )


@ex.capture
def get_dataset(data):
    # _log.info("Opening xarray dataset")
    try:
        dataset = xr.open_zarr(data)
    except ValueError:
        dataset = xr.open_dataset(data)

    try:
        return dataset.isel(step=0).drop('step').drop('p')
    except:
        return dataset


@ex.capture
def get_data_loader(data: xr.Dataset, train, training_slices,
                    validation_slices, prognostics, batch_size):

    from torch.utils.data.dataloader import default_collate
    from uwnet.timestepper import Batch

    if train:
        slices = training_slices
    else:
        slices = validation_slices

    ds = data.isel(
        y=slice(*slices['y']),
        x=slice(*slices['x']),
        time=slice(*slices['time']))

    def my_collate_fn(batch):
        return Batch(default_collate(batch), prognostics)

    train_data = XRTimeSeries(ds)

    if len(train_data) == 0:
        raise ValueError("No data available. Are any of the slices in "
                         "training_slices or validation_slices length 0.")
    return DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=my_collate_fn)


def get_validation_engine(model, dt):
    def _validate(engine, batch):
        return get_input_output(model, dt, batch)

    return Engine(_validate)


@ex.capture
def get_output_dir(_run=None, model_dir=None, output_dir=None):
    """Get a unique output directory name using the run ID that sacred
    assigned OR return the specified output directory
    """
    if output_dir:
        return output_dir
    else:
        file_name = str(_run._id)
        return join(model_dir, file_name)


def is_one_dimensional(val):
    return val.dim() == 2


class Trainer(object):
    """Utility object for training a neural network parametrization
    """

    @ex.capture
    def __init__(self, _run, lr, loss_scale):
        # setup logging
        logging.basicConfig(level=logging.INFO)

        self.compute_metrics = False

        self.logger = logging.getLogger(__name__)

        # get output directory
        self.output_dir = get_output_dir()

        self.dataset = get_dataset()
        self.mass = torch.tensor(self.dataset.layer_mass.values).view(
            -1, 1, 1).float()
        self.z = torch.tensor(self.dataset.z.values).float()
        self.time_step = get_timestep(self.dataset)
        self.train_loader = get_data_loader(self.dataset, train=True)
        self.test_loader = get_data_loader(self.dataset, train=False)
        self.model = get_model(*get_pre_post(self.dataset, self.train_loader))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = weighted_mean_squared_error(
            weights=self.mass / self.mass.mean(), dim=-3)
        self.plot_manager = TrainingPlotManager(ex, self.model, self.dataset)
        self.setup_validation_engine()
        self.setup_engine()

    def log_validation_results(self, trainer):
        self.tester.run(self.test_loader)
        metrics = self.tester.state.metrics

        log_str = "test metrics: "
        for name, val in metrics.items():
            ex.log_scalar("train_" + name, val)
            log_str += f'{name}: {val:.2f}\t'
        self.logger.info(log_str)

        metrics = trainer.state.metrics
        log_str = "train metrics: "
        for name, val in metrics.items():
            ex.log_scalar("test_" + name, val)
            log_str += f'{name}: {val:.2f}\t'
        self.logger.info(log_str)

    def setup_validation_engine(self):
        self.tester = get_validation_engine(self.model, self.time_step)
        self.setup_metrics_for_engine(self.tester)

    @ex.capture(prefix='step')
    def get_step(self, name, kwargs, _log):
        _log.info(f"Using `{name}` gradient stepper")
        if name == 'multi':
            self.logger.info(f"Disabling computation of metrics")
            self.compute_metrics = False
        else:
            self.compute_metrics = True
        return get_step(name, self, kwargs)

    def setup_engine(self):
        step = self.get_step()
        self.engine = Engine(step)
        self.setup_metrics_for_engine(self.engine)

        self.engine.add_event_handler(Events.ITERATION_COMPLETED,
                                      self.after_batch)
        self.engine.add_event_handler(Events.ITERATION_COMPLETED,
                                      self.print_loss_info)
        self.engine.add_event_handler(Events.EPOCH_COMPLETED, self.after_epoch)
        self.engine.add_event_handler(Events.EPOCH_COMPLETED,
                                      self.log_validation_results)

    @ex.capture
    def setup_metrics_for_engine(self, engine, prognostics):
        from .metrics import WeightedMeanSquaredError

        if not self.compute_metrics:
            return

        @curry
        def output_transform(key, args):
            x, y = args
            return x[key], y[key]

        for key in prognostics:
            metric = WeightedMeanSquaredError(
                self.mass, output_transform=output_transform(key))

            metric.attach(engine, key)

    def print_loss_info(self, engine):
        n = len(self.train_loader)
        batch = engine.state.iteration % (n + 1)
        log_str = f"[{batch}/{n}]:\t"
        for key, val in engine.state.loss_info.items():
            log_str += f'{key}: {val:.2f}\t'
            ex.log_scalar(key, val)
        self.logger.info(log_str)

    def after_batch(self, engine):
        state = engine.state
        batch_info = {
            'epoch': state.epoch,
            'loss': state.output,
        }
        ex.log_scalar('loss', batch_info['loss'])

    def after_epoch(self, engine):
        # save artifacts
        n = engine.state.epoch
        epoch_file = f"{n}.pkl"

        with self.change_to_work_dir():
            torch.save(self.model, epoch_file)
            ex.add_artifact(epoch_file)
            self.plot_manager(engine)

    def _make_work_dir(self):
        try:
            os.makedirs(self.output_dir)
        except OSError:
            pass

    @contextmanager
    def change_to_work_dir(self):
        """Context manager for using a working directory"""
        self.logger.info(f"Saving outputs in {self.output_dir}")
        self._make_work_dir()
        try:
            cwd = os.getcwd()
            os.chdir(self.output_dir)
            yield
        finally:
            os.chdir(cwd)

    @ex.capture
    def train(self, epochs):
        """Train the neural network for a fixed number of epochs"""
        self.engine.run(self.train_loader, max_epochs=epochs)


@ex.command()
def train_pre_post(prepost):
    """Train the pre and post processing modules"""
    dataset = get_dataset()
    logging.info(f"Saving Pre/Post module to {prepost['path']}")
    torch.save(get_pre_post(dataset), prepost['path'])


@ex.automain
def main():
    Trainer().train()
