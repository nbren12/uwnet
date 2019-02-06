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
from os.path import join
from functools import partial

from sacred import Experiment

import torch
import xarray as xr
from torch.utils.data import DataLoader
from stochastic_parameterization.utils import (
    get_xarray_dataset_with_eta,
)
from uwnet.model import get_model
from uwnet.pre_post import get_pre_post
from uwnet.training_plots import TrainingPlotManager
from uwnet.loss import (weighted_mean_squared_error, get_step)
from uwnet.datasets import XRTimeSeries, ConditionalXRSampler, get_timestep
from ignite.engine import Engine, Events

ex = Experiment("Q1")

XRTimeSeries = ex.capture(XRTimeSeries)
TrainingPlotManager = ex.capture(TrainingPlotManager, prefix='plots')
get_model = ex.capture(get_model, prefix='model')
get_pre_post = ex.capture(get_pre_post, prefix='prepost')
get_step = ex.capture(get_step)


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
    binning_quantiles = None
    eta_to_train = None
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
    y = (None, None)
    x = (None, None)
    time_sl = (None, None)
    output_dir = None

    prognostics = ['QT', 'SLI']
    prepost = dict(
        kind='pca',
        path='models/prepost.pkl'
    )

    model = dict(
        kind='inner_model'
    )

    plots = dict(
        interval=1,
        single_column_locations=[(32, 0)]
    )
    step_type = 'instability'
    binning_method = 'precip'
    base_model_location = None


@ex.capture
def get_xarray_dataset(
        data, binning_quantiles, binning_method, base_model_location):
    return get_xarray_dataset_with_eta(
        data, binning_quantiles, binning_method, base_model_location
    )


@ex.capture
def get_data_loader(data: xr.Dataset, x, y, time_sl, vertical_grid_size,
                    batch_size, prognostics, eta_to_train):
    prognostics = ['QT', 'SLI']
    from torch.utils.data.dataloader import default_collate
    from uwnet.timestepper import Batch
    ds = data.isel(
        z=slice(0, vertical_grid_size),
        y=slice(*y),
        x=slice(*x),
        time=slice(*time_sl))

    def my_collate_fn(batch):
        return Batch(default_collate(batch), prognostics)

    train_data = XRTimeSeries(ds)
    print(eta_to_train)
    if eta_to_train is not None:
        train_data = ConditionalXRSampler(ds, eta_to_train)
    else:
        train_data = XRTimeSeries(ds)
    return DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=my_collate_fn)


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
        # db = MongoDBLogger()
        # experiment = Experiment(api_key="fEusCnWmzAtmrB0FbucyEggW2")
        self.logger = logging.getLogger(__name__)

        # get output directory
        self.output_dir = get_output_dir()
        print(self.output_dir)
        self.dataset = get_xarray_dataset()
        self.mass = torch.tensor(self.dataset.layer_mass.values).view(
            -1, 1, 1).float()
        self.z = torch.tensor(self.dataset.z.values).float()
        self.time_step = get_timestep(self.dataset)
        self.train_loader = get_data_loader(self.dataset)
        self.model = get_model(*get_pre_post(self.dataset))
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=lr)
        self.criterion = weighted_mean_squared_error(
            weights=self.mass / self.mass.mean(), dim=-3)
        self.plot_manager = TrainingPlotManager(ex, self.model, self.dataset)
        self.setup_engine()

    def setup_engine(self):
        step = partial(get_step(), self)
        self.engine = Engine(step)
        self.engine.add_event_handler(
            Events.ITERATION_COMPLETED, self.after_batch)
        self.engine.add_event_handler(
            Events.ITERATION_COMPLETED, self.print_loss_info)
        self.engine.add_event_handler(
            Events.EPOCH_COMPLETED, self.after_epoch)

    def print_loss_info(self, engine):
        n = len(self.train_loader)
        batch = engine.state.iteration % (n + 1)
        log_str = f"[{batch}/{n}]:\t"
        for key, val in engine.state.loss_info.items():
            log_str += f'{key}: {val:.2f}\t'
            ex.log_scalar(key, val)
        self.logger.info(log_str)

    def compute_source_r2(self, batch):
        from .timestepper import Batch
        from toolz import merge_with
        from .loss import weighted_r2_score, r2_score
        src = self.model(batch)

        # compute the apparent source
        batch = Batch(batch, self.prognostics)
        g = batch.get_known_forcings()
        progs = batch.data[self.prognostics]
        storage = progs.apply(lambda x: (x[1:] - x[:-1]) / self.time_step)
        forcing = g.apply(lambda x: (x[1:] + x[:-1]) / 2)
        src = src.apply(lambda x: (x[1:] + x[:-1]) / 2)
        true_src = storage - forcing * 86400

        # copmute the metrics
        def wr2_score(args):
            x, y = args
            return weighted_r2_score(x, y, self.mass, dim=-3).item()

        r2s = merge_with(wr2_score, true_src, src)
        print(r2s)

        # compute the r2 of the integral
        pred_int = src.apply(lambda x: (x * self.mass).sum(-3))
        true_int = true_src.apply(lambda x: (x * self.mass).sum(-3))

        def scalar_r2_score(args):
            return r2_score(*args).item()

        def bias(args):
            x, y = args
            return (y.mean() - x.mean()).item() / 1000

        r2s = merge_with(scalar_r2_score, true_int, pred_int)
        print(r2s)

        r2s = merge_with(bias, true_int, pred_int)
        print(r2s)

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
        print('training')
        self.engine.run(self.train_loader, max_epochs=epochs)


@ex.command()
def train_pre_post(prepost):
    """Train the pre and post processing modules"""
    dataset = get_xarray_dataset()
    logging.info(f"Saving Pre/Post module to {prepost['path']}")
    torch.save(get_pre_post(dataset), prepost['path'])


@ex.automain
def main():
    Trainer().train()
