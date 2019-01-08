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
import numpy as np
from contextlib import contextmanager
from os.path import join

from sacred import Experiment

import torch
import xarray as xr
from torch.utils.data import DataLoader
from uwnet.model import get_model
from uwnet.pre_post import get_pre_post
from uwnet.training_plots import TrainingPlotManager
from uwnet.datasets import XRTimeSeries, ConditionalXRSampler, get_timestep
from uwnet.loss import (weighted_mean_squared_error, total_loss)
from ignite.engine import Engine, Events

ex = Experiment("Q1")

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
    seq_length = 1
    batch_size = 256
    vertical_grid_size = 34
    precip_quantiles = None
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


@ex.capture
def set_eta(dataset, precip_quantiles):
    if not precip_quantiles:
        return dataset
    bins = [
        dataset.Prec.quantile(quantile).values
        for quantile in precip_quantiles
    ]
    eta_ = dataset['Prec'].copy()
    eta_.values = np.digitize(dataset.Prec.values, bins, right=True)
    dataset['eta'] = eta_
    dataset['eta'].attrs['units'] = 'N/A'
    dataset['eta'].attrs['long_name'] = 'Stochastic State'
    return dataset


@ex.capture
def get_xarray_dataset(data, precip_quantiles):
    try:
        dataset = xr.open_zarr(data)
    except ValueError:
        dataset = xr.open_dataset(data)

    dataset = set_eta(dataset)
    try:
        return dataset.isel(step=0).drop('step').drop('p')
    except:
        return dataset


@ex.capture
def get_data_loader(data: xr.Dataset, x, y, time_sl, vertical_grid_size,
                    batch_size, eta_to_train):
    ds = data.isel(
        z=slice(0, vertical_grid_size),
        y=slice(*y),
        x=slice(*x),
        time=slice(*time_sl))

    if eta_to_train:
        train_data = ConditionalXRSampler(ds, eta_to_train)
    else:
        train_data = XRTimeSeries(ds)
    return DataLoader(
        train_data, batch_size=batch_size, shuffle=True)


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


@ex.capture
def get_model_inputs_from_batch(batch, prognostics):
    """Redimension a batch from a torch data loader

    Torch's data loader class is very helpful, but it produces data which has a shape of (batch, feature). However, the models require input in the physical dimensions (time, z, y, x), this function reshapes these arrays.
    """
    from .timestepper import Batch
    return Batch(batch, prognostics)


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

        self.dataset = get_xarray_dataset()
        self.mass = torch.tensor(self.dataset.layer_mass.values).view(
            -1, 1, 1).float()
        self.z = torch.tensor(self.dataset.z.values).float()
        self.time_step = get_timestep(self.dataset)
        self.train_loader = get_data_loader(self.dataset)

        self.model = get_model(*get_pre_post(self.dataset))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = weighted_mean_squared_error(
            weights=self.mass / self.mass.mean(), dim=-3)
        self.plot_manager = TrainingPlotManager(ex, self.model, self.dataset)
        self.setup_engine()

    def setup_engine(self):
        self.engine = Engine(self.step)
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

    def step(self, engine, batch):
        self.optimizer.zero_grad()
        batch = get_model_inputs_from_batch(batch)
        loss, loss_info = total_loss(
            self.criterion, self.model, self.z, batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        engine.state.loss_info = loss_info
        return loss_info

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
