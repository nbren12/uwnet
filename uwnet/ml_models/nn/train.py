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
import matplotlib
matplotlib.use('agg')
import logging
import os
from contextlib import contextmanager
from os.path import join

import xarray as xr
from sacred import Experiment
from toolz import curry

import torch
from torch.optim.lr_scheduler import StepLR
from ignite.engine import Engine, Events
from .datasets_handler import get_timestep, XarrayBatchLoader, get_dataset, get_data_loader
from uwnet.loss import get_input_output, get_step, weighted_mean_squared_error
from uwnet.model import get_model
from uwnet.pre_post import get_pre_post
from uwnet.training_plots import TrainingPlotManager
from uwnet.metrics import WeightedMeanSquaredError

ex = Experiment("Q1", interactive=True)

TrainingPlotManager = ex.capture(TrainingPlotManager, prefix='plots')

get_model = ex.capture(get_model, prefix='model')
get_pre_post = ex.capture(get_pre_post, prefix='prepost')
get_dataset = ex.capture(get_dataset)
get_data_loader = ex.capture(get_data_loader)


@ex.config
def my_config():
    """Default configurations managed by sacred"""
    train_data = ""
    test_data = ""

    predict_radiation = True
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

    output_dir = None

    prognostics = ['QT', 'SLI']
    prepost = dict(kind='pca', path='models/prepost.pkl')

    model = dict(kind='inner_model')

    plots = dict(interval=1, single_column_locations=[(32, 0)])
    step = dict(
        name='instability',
        kwargs={'alpha': 1.0}
    )
    lr_decay_rate = None
    lr_step_size = 5


def get_plot_manager(model):
    from src.data import open_data
    dataset = open_data('training')
    return TrainingPlotManager(ex, model, dataset)


def get_validation_engine(model, dt):
    def _validate(engine, data):
        from uwnet.timestepper import Batch
        # TODO this code duplicaates loss.py:107
        batch = Batch(data.float(), prognostics=['QT', 'SLI'])
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


class Trainer(object):
    """Utility object for training a neural network parametrization
    """

    @ex.capture
    def __init__(self, _run, lr, loss_scale, train_data, test_data,
                 lr_decay_rate=None, lr_step_size=5):
        # setup logging
        logging.basicConfig(level=logging.INFO)

        self.compute_metrics = False

        self.logger = logging.getLogger(__name__)

        # get output directory
        self.output_dir = get_output_dir()

        train_dataset = get_dataset(train_data)
        test_dataset = get_dataset(test_data)

        self.mass = torch.tensor(train_dataset.layer_mass.values).view(
            -1, 1, 1).float()
        self.z = torch.tensor(train_dataset.z.values).float()
        self.time_step = get_timestep(train_dataset)
        self.train_loader = get_data_loader(train_dataset)
        self.test_loader = get_data_loader(test_dataset)
        self.model = get_model(*get_pre_post(train_dataset, self.train_loader))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if lr_decay_rate is None:
            self.lr_scheduler = None
        else:
            self.lr_scheduler = StepLR(
                self.optimizer, step_size=lr_step_size, gamma=lr_decay_rate)

        self.criterion = weighted_mean_squared_error(
            weights=self.mass / self.mass.mean(), dim=-3)
        self.plot_manager = get_plot_manager(self.model)
        self.setup_validation_engine()
        self.setup_engine()

    def step_lr_scheduler(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

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

        self.step_lr_scheduler()

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
