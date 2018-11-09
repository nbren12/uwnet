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
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from toolz import valmap

import torch
import torchnet as tnt
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader
from uwnet import model
from uwnet.columns import single_column_simulation
from uwnet.datasets import XRTimeSeries
from uwnet.loss import compute_multiple_step_loss, weighted_mean_squared_error, mse_with_integral

ex = Experiment("Q1")


def get_dataset(data):
    try:
        dataset = xr.open_zarr(data)
    except ValueError:
        dataset = xr.open_dataset(data)

    try:
        return dataset.isel(step=0).drop('step').drop('p')
    except:
        return dataset


def water_budget_plots(model, ds, location, filenames):
    nt = min(len(ds.time), 190)
    scm_data = single_column_simulation(model, location, interval=(0, nt - 1))
    merged_pred_data = location.rename({
        'SLI': 'SLIOBS',
        'QT': 'QTOBS'
    }).merge(
        scm_data, join='inner')
    output = model.call_with_xr(merged_pred_data)

    plt.figure()
    scm_data.QT.plot(x='time')
    plt.title("QT")
    plt.savefig(filenames[0])

    plt.figure()
    output.QT.plot(x='time')
    plt.title("FQTNN from NN-prediction")
    output_truth = model.call_with_xr(location)
    plt.savefig(filenames[1])

    plt.figure()
    output_truth.QT.plot(x='time')
    plt.title("FQTNN from Truth")
    plt.xlim([100, 125])
    plt.savefig(filenames[2])

    plt.figure()
    (scm_data.QT * ds.layer_mass / 1000).sum('z').plot()
    (location.QT * ds.layer_mass / 1000).sum('z').plot(label='observed')
    plt.legend()
    plt.savefig(filenames[3])


## Some plotting routines to be called after every epoch
class Plot(object):
    def get_filename(self, epoch):
        return self.name.format(epoch)

    def save_figure(self, epoch, location, output):
        fig, ax = plt.subplots()
        self.plot(location, output, ax)
        path = self.get_filename(epoch)
        logging.info(f"Saving to {path}")
        fig.savefig(path)
        ex.add_artifact(path)
        plt.close()


class plot_q2(Plot):
    name = 'q2_{}.png'

    def plot(self, location, output, ax):
        return output.QT.plot(x='time', ax=ax)


class plot_scatter_q2_fqt(Plot):
    name = 'scatter_fqt_q2_{}.png'

    def plot(self, location, output, ax):
        x, y, z = [
            x.values.ravel()
            for x in xr.broadcast(location.FQT * 86400, output.QT, output.z)
        ]
        im = ax.scatter(x, y, c=z)
        plt.colorbar(im, ax=ax)


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


@ex.config
def my_config():
    """Default configurations managed by sacred"""
    restart = False
    lr = .001
    epochs = 2
    model_dir = 'models'
    skip = 5
    seq_length = 1
    batch_size = 256
    tag = ''
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
    y = (None, None)
    x = (None, None)
    time_sl = (None, None)
    min_output_interval = 0
    output_dir = None


def is_one_dimensional(val):
    return val.dim() == 2


def redimension_torch_loader_output(val):
    if is_one_dimensional(val):
        return val.t().unsqueeze(-1)
    else:
        return val.permute(1, 2, 0).unsqueeze(-1)


def get_model_inputs_from_batch(batch):
    """Redimension a batch from a torch data loader

    Torch's data loader class is very helpful, but it produces data which has a shape of (batch, feature). However, the models require input in the physical dimensions (time, z, y, x), this function reshapes these arrays.
    """
    return valmap(redimension_torch_loader_output, batch)


class Trainer(object):
    """Utility object for training a neural network parametrization

    Attributes
    ----------
    model
    logger
    meters
        a dictionary of the torchnet meters used for tracking things like
        training losses
    output_dir: str
        the output directory
    criterion
        the loss function

    Methods
    -------
    train

    """

    @ex.capture
    def __init__(self, _run, restart, lr, batch_size, tag, data,
                 vertical_grid_size, loss_scale, y, x, time_sl,
                 min_output_interval):
        # setup logging
        logging.basicConfig(level=logging.INFO)

        # db = MongoDBLogger()
        # experiment = Experiment(api_key="fEusCnWmzAtmrB0FbucyEggW2")
        self.logger = logging.getLogger(__name__)

        self.min_output_interval = min_output_interval

        # get output directory
        self.output_dir = get_output_dir()

        # set up meters
        self.meters = dict(loss=tnt.meter.AverageValueMeter())

        self.dataset = get_dataset(data)
        self.mass = torch.tensor(self.dataset.layer_mass.values).view(
            -1, 1, 1).float()

        ds = self.dataset.isel(
            z=slice(0, vertical_grid_size),
            y=slice(*y),
            x=slice(*x),
            time=slice(*time_sl))

        self.nt = len(ds.time)

        train_data = XRTimeSeries(ds)
        self.train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True)
        self.constants = train_data.torch_constants()

        # compute standard deviation
        self.logger.info("Computing Standard Deviation")
        scale = train_data.scale

        # compute scaler
        self.logger.info("Computing Mean")
        mean = train_data.mean

        input_fields = (('QT', vertical_grid_size),
                        ('SLI', vertical_grid_size), ('SST', 1), ('SOLIN', 1))
        output_fields = (('QT', vertical_grid_size), ('SLI',
                                                      vertical_grid_size))

        self.prognostics = ['QT', 'SLI']
        self.time_step = float(train_data.timestep())

        # initialize model
        self.model = model.ApparentSource(
            mean, scale, inputs=input_fields, outputs=output_fields)

        # initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # set up loss function
        self.criterion = weighted_mean_squared_error(weights=self.mass/self.mass.mean(), dim=-3)
        # self.criterion = mse_with_integral(weights=self.mass, dim=-3)
        self.epoch = 0

    def _increment_step_count(self):
        try:
            self.step_count += 1
        except AttributeError:
            self.step_count = 1

    def before_batch(self):
        self._time_batch_start = time()

    def _compute_loss(self, batch, initial_time, seq_length):
        return compute_multiple_step_loss(self.criterion, self.model, batch,
                                          self.prognostics, initial_time,
                                          seq_length, self.time_step)

    @ex.capture
    def _train_with_batch(self, k, batch, seq_length, skip):
        logging.info(f"Batch {k} of {len(self.train_loader)}")
        self.before_batch()
        self._increment_step_count()
        self.optimizer.zero_grad()
        loss = self._compute_loss(batch, initial_time=0, seq_length=0)
        loss.backward()
        self.optimizer.step()
        self.meters['loss'].add(loss.item())
        self.compute_source_r2(batch)
        self.after_batch()

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

    def after_batch(self):
        time_elapsed_batch = time() - self._time_batch_start
        batch_info = {
            'epoch': self.epoch,
            'loss': self.meters['loss'].value()[0],
            'time_elapsed': time_elapsed_batch,
        }

        ex.log_scalar('loss', batch_info['loss'])
        ex.log_scalar('time_elapsed', batch_info['time_elapsed'])
        self.meters['loss'].reset()
        self.logger.info(f"Loss: {batch_info['loss']}; "
                         f"Time Elapsed {time_elapsed_batch} ")

    def _train_for_epoch(self):
        self.epoch += 1
        logging.info(f"Epoch {self.epoch}")
        for k, batch in enumerate(self.train_loader):
            input_data = get_model_inputs_from_batch(batch)
            self._train_with_batch(k, input_data)
        self.after_epoch()

    def after_epoch(self):
        # save artifacts
        if self.step_count > self.min_output_interval:
            epoch_file = f"{self.epoch}.pkl"
            torch.save(self.model, epoch_file)
            ex.add_artifact(epoch_file)
            self._after_epoch_plots()
            self.step_count = 0

    def _after_epoch_plots(self):
        single_column_plots = [plot_q2(), plot_scatter_q2_fqt()]
        for y in [32]:
            location = self.dataset.isel(y=slice(y, y + 1), x=slice(0, 1), time=slice(0, 200))
            output = self.model.call_with_xr(location)
            for plot in single_column_plots:
                plot.save_figure(f'{self.epoch}-{y}', location, output)

            i = self.epoch
            filenames = [
                name + f'{i}-{y}'
                for name in ['qt', 'fqtnn', 'fqtnn-obs', 'pw']
            ]
            water_budget_plots(self.model, self.dataset, location, filenames)

    def _make_work_dir(self):
        try:
            os.makedirs(self.output_dir)
        except OSError:
            pass

    @contextmanager
    def _change_to_work_dir(self):
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
        with self._change_to_work_dir():
            for i in range(epochs):
                self._train_for_epoch()


@ex.automain
def main():
    Trainer().train()
