import logging
import os
from os.path import join
from time import time

from sacred import Experiment

from toolz import valmap
import torch
import torchnet as tnt
import xarray as xr
from torch.utils.data import DataLoader
from torch import nn
from uwnet import model
from uwnet.datasets import XRTimeSeries
from uwnet.utils import batch_to_model_inputs, select_time
from uwnet.loss import compute_multiple_step_loss
import matplotlib.pyplot as plt

ex = Experiment("Q1")


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


def after_epoch(epoch, model, dataset, y=[5, 15, 32]):
    from uwnet.model import call_with_xr

    single_column_plots = [plot_q2(), plot_scatter_q2_fqt()]
    for y in [5, 15, 32]:
        location = dataset.isel(y=slice(y, y+1), x=slice(0, 1))
        output = call_with_xr(model, location, drop_times=0)
        for plot in single_column_plots:
            plot.save_figure(f'{epoch}-{y}', location, output)


@ex.capture
def get_output_dir(_run=None, base='.trained_models'):
    """Get a unique output directory name"""
    file_name = str(_run._id)
    return join(base, file_name)


@ex.config
def my_config():
    restart = False
    lr = .001
    n_epochs = 2
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


def is_one_dimensional(val):
    return val.dim() == 2


def redimension_torch_loader_output(val):
    if is_one_dimensional(val):
        return val.t().unsqueeze(-1)
    else:
        return val.permute(1, 2, 0).unsqueeze(-1)


def get_model_inputs_from_batch(batch):
    return valmap(redimension_torch_loader_output, batch)


@ex.automain
def main(_run, restart, lr, n_epochs, model_dir, skip, seq_length, batch_size,
         tag, data, vertical_grid_size, loss_scale, y, x, time_sl,
         min_output_interval):
    # setup logging
    logging.basicConfig(level=logging.INFO)

    # db = MongoDBLogger()
    # experiment = Experiment(api_key="fEusCnWmzAtmrB0FbucyEggW2")
    logger = logging.getLogger(__name__)

    # get output directory
    output_dir = get_output_dir(base=model_dir)

    # switch to output directory
    logger.info(f"Saving outputs in {output_dir}")
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    # set up meters
    meter_loss = tnt.meter.AverageValueMeter()

    # get training loader
    def post(x):
        return x

    logger.info("Opening Training Data")
    try:
        dataset = xr.open_zarr(data)
    except ValueError:
        dataset = xr.open_dataset(data)

    ds = dataset.isel(
        z=slice(0, vertical_grid_size),
        y=slice(*y),
        x=slice(*x),
        time=slice(*time_sl))
    nt = len(ds.time)

    train_data = XRTimeSeries(ds)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    constants = train_data.torch_constants()

    # compute standard deviation
    logger.info("Computing Standard Deviation")
    scale = train_data.scale

    # compute scaler
    logger.info("Computing Mean")
    mean = train_data.mean

    prognostics = ['QT', 'SLI']
    input_fields = (('QT', vertical_grid_size), ('SLI', vertical_grid_size),
                    ('SST', 1), ('SOLIN', 1))
    output_fields = (('QT', vertical_grid_size), ('SLI', vertical_grid_size))

    # initialize model
    lstm = model.ApparentSource(
        mean, scale, inputs=input_fields, outputs=output_fields)

    dt = float(train_data.timestep())

    # initialize optimizer
    optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

    # set up loss function
    criterion = nn.MSELoss()

    steps_out = 0

    os.chdir(output_dir)
    try:
        for i in range(n_epochs):
            logging.info(f"Epoch {i}")
            for k, batch in enumerate(train_loader):
                steps_out += 1
                logging.info(f"Batch {k} of {len(train_loader)}")
                time_batch_start = time()

                batch = get_model_inputs_from_batch(batch)
                for initial_time in range(0, nt - seq_length, skip):
                    optimizer.zero_grad()
                    loss = compute_multiple_step_loss(criterion, lstm, batch,
                                                      prognostics,
                                                      initial_time, seq_length,
                                                      dt)
                    loss.backward()

                    # take step
                    optimizer.step()

                    # Log the results
                    meter_loss.add(loss.item())

                time_elapsed_batch = time() - time_batch_start

                batch_info = {
                    'epoch': i,
                    'batch': k,
                    'loss': meter_loss.value()[0],
                    'time_elapsed': time_elapsed_batch,
                }

                ex.log_scalar('loss', batch_info['loss'])
                ex.log_scalar('time_elapsed', batch_info['time_elapsed'])

                logger.info(f"Batch {k},  Loss: {meter_loss.value()[0]}; "
                            f"Time Elapsed {time_elapsed_batch} ")

            # save artifacts
            if steps_out > min_output_interval:
                print("Saving")
                steps_out = 0
                epoch_file = f"{i}.pkl"
                torch.save(lstm, epoch_file)
                ex.add_artifact(epoch_file)
                after_epoch(i, lstm, dataset)

    except KeyboardInterrupt:
        torch.save({'epoch': i, 'dict': lstm.to_dict()}, "interrupt.pkl")
