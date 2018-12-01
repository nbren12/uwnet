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

import matplotlib.pyplot as plt
from sacred import Experiment
from toolz import valmap

import torch
import torch.nn.functional as F
import torchnet as tnt
import xarray as xr
from torch.utils.data import DataLoader
from uwnet import model
from uwnet.model import get_model
from uwnet.columns import single_column_simulation
from uwnet.datasets import XRTimeSeries
from uwnet.loss import (weighted_mean_squared_error, total_loss)
from ignite.engine import Engine, Events

ex = Experiment("Q1")

@ex.capture
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
    prognostics=['QT', 'SLI']
    scm_data = single_column_simulation(model, location, interval=(0, nt - 1),
                                        prognostics=prognostics)
    merged_pred_data = location.rename({
        'SLI': 'SLIOBS',
        'QT': 'QTOBS',
        'U': 'UOBS',
        'V': 'VOBS',
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
    data = "data/processed/training.nc"
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

    prognostics = ['QT', 'SLI']


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
    def __init__(self, _run, restart, lr, batch_size, tag, vertical_grid_size,
                 loss_scale, y, x, time_sl, min_output_interval):
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

        self.dataset = get_dataset()
        self.mass = torch.tensor(self.dataset.layer_mass.values).view(
            -1, 1, 1).float()
        self.z = torch.tensor(self.dataset.z.values).float()

        self.dataset_subset = self.dataset.isel(
            z=slice(0, vertical_grid_size),
            y=slice(*y),
            x=slice(*x),
            time=slice(*time_sl))

        ds =  self.dataset_subset

        self.nt = len(ds.time)

        train_data = XRTimeSeries(ds, time_length=160)
        self.train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True)
        self.constants = train_data.torch_constants()

        # compute standard deviation
        self.logger.info("Computing Standard Deviation")
        self.scale = train_data.scale

        # compute scaler
        self.logger.info("Computing Mean")
        self.mean = train_data.mean

        self.time_step = float(train_data.timestep())
        self.setup_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = weighted_mean_squared_error(
            weights=self.mass / self.mass.mean(), dim=-3)
        self.plot_interval = 1
        self.setup_engine()

    @ex.capture
    def setup_model(self, vertical_grid_size):
        self.model = get_model(self.mean, self.scale, vertical_grid_size)

    def setup_engine(self):
        self.engine = Engine(self.step)
        self.engine.add_event_handler(
            Events.ITERATION_COMPLETED, self.after_batch)
        self.engine.add_event_handler(
            Events.ITERATION_COMPLETED, self.print_loss_info)
        self.engine.add_event_handler(
            Events.EPOCH_COMPLETED, self.after_epoch)
        self.engine.add_event_handler(
            Events.EPOCH_COMPLETED, self.imbalance_plot)

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

        n = len(self.train_loader)
        batch = state.iteration % (n + 1)
        ex.log_scalar('loss', batch_info['loss'])

    def after_epoch(self, engine):
        # save artifacts
        n = engine.state.epoch
        epoch_file = f"{n}.pkl"

        if n % self.plot_interval != 0:
            return

        with self.change_to_work_dir():
            torch.save(self.model, epoch_file)
            ex.add_artifact(epoch_file)
            self.plot_model(engine)

    def plot_model(self, engine):
        single_column_plots = [plot_q2(), plot_scatter_q2_fqt()]
        for y in [32]:
            location = self.dataset.isel(
                y=slice(y, y + 1), x=slice(0, 1), time=slice(0, 200))
            output = self.model.call_with_xr(location)
            for plot in single_column_plots:
                plot.save_figure(f'{engine.state.epoch}-{y}', location, output)

            i = engine.state.epoch
            filenames = [
                name + f'{i}-{y}'
                for name in ['qt', 'fqtnn', 'fqtnn-obs', 'pw']
            ]
            water_budget_plots(self.model, self.dataset, location, filenames)

    def imbalance_plot(self, engine):
        from uwnet.thermo import lhf_to_evap
        n = engine.state.epoch
        if n % self.plot_interval != 0:
            return
        mass = self.dataset.layer_mass
        subset = self.dataset_subset.isel(time=slice(0, None, 20))
        out = self.model.call_with_xr(subset)
        pme = (subset.Prec- lhf_to_evap(subset.LHF)).mean(['time', 'x'])
        pmenn = - (mass * out.QT).sum('z')/1000
        pmefqt = ((mass * subset.FQT).sum('z')/1000*86400).mean(['time', 'x'])
        pmenn = pmenn.mean(['time', 'x'])
        plotme = xr.Dataset({'truth': pme, 'nn': pmenn, 'fqt':
                            pmefqt}).to_array(dim='var')

        plt.figure()
        plotme.plot(hue='var')

        with self.change_to_work_dir():
            plt.savefig(f"{n}-imbalance.png")

        plt.close()

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


@ex.automain
def main():
    Trainer().train()
