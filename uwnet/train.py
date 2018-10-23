import logging
import os
import re
from os.path import join
from time import time

import matplotlib.pyplot as plt
from sacred import Experiment
from toolz import merge

import torch
import torchnet as tnt
import xarray as xr
from torch.utils.data import DataLoader
from uwnet import criticism, model
from uwnet.model import call_with_xr
from uwnet.datasets import XRTimeSeries
from uwnet.loss import MVLoss
from uwnet.utils import batch_to_model_inputs, select_time

ex = Experiment()


def after_epoch(model, dataset, i):
    """Tasks to perform after an epoch is complete

    These include making plots of the performance. Saving files

    Yields
    ------
    artifacts : str
        list of paths to saved artifacts
    """
    water_imbalance_pth = f"water_imbalance_{i}.pdf"
    plt.figure()
    # criticism.plot_water_imbalance(dataset, model)
    plt.savefig(water_imbalance_pth)
    yield water_imbalance_pth

    # compute
    output =call_with_xr(model, dataset)
    fqtnn_path = f"fqtnn_{i}.png"
    plt.figure()
    (86400 * output.FQTNN.isel(x=0, y=0)).plot(x='time')
    plt.savefig(fqtnn_path)
    yield fqtnn_path

    # compute
    time = dataset.time
    dt = float(time[1] - time[0])
    q2 = dataset.QT.diff('time') / dt - dataset.FQT * 86400
    q2_path = f"q2_{i}.png"
    plt.figure()
    q2.isel(x=0, y=0).plot(x='time')
    plt.savefig(q2_path)
    yield q2_path
    plt.close()

    # x, y, c = [
    #     x.values.ravel() for x in xr.broadcast(q2, 86400 * output.FQTNN, q2.z)
    # ]

    # scatter_path = f"scatter_{i}.png"
    # from matplotlib.colors import LogNorm
    # plt.figure()
    # plt.scatter(x, y, c=c, norm=LogNorm(), rasterized=True, alpha=.1)
    # plt.xlabel("Q2 Obs", "FQTNN")
    # plt.xlim([a, b])
    # a = min(x.min(), y.min())
    # b = max(x.max(), y.max())
    # plt.xlim([a, b])
    # plt.ylim([a, b])
    # plt.plot([a,b], [a,b], c='k')
    # plt.colorbar()
    # plt.savefig(scatter_path)
    # yield scatter_path
    # plt.close('all')


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
    seq_length = 10
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
    auxiliary = [('LHF', 1), ('SHF', 1)]
    prognostic = [('QT', 34), ('SLI', 34)]
    diagnostic = []
    forcing = ['QT', 'SLI']

    min_output_interval = 0

@ex.automain
def main(_run, auxiliary, prognostic, diagnostic, forcing, restart,
         lr, n_epochs, model_dir, skip, seq_length, batch_size, tag, data,
         vertical_grid_size, loss_scale, y, x, time_sl, min_output_interval):
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

    # experiment.log_parameter('directory', os.path.abspath(output_dir))
    n_epochs = n_epochs
    batch_size = batch_size
    seq_length = seq_length

    # set up meters
    meter_loss = tnt.meter.AverageValueMeter()
    meter_avg_loss = tnt.meter.AverageValueMeter()

    # get training loader
    def post(x):
        return x

    logger.info("Opening Training Data")
    try:
        ds = xr.open_zarr(data)
    except ValueError:
        ds = xr.open_dataset(data)

    ds = ds.isel(
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

    cls = model.ForcedStepper

    logger.info(f"Training with {cls}")

    # restart
    if restart:
        path = os.path.abspath(restart)
        logger.info(f"Restarting from checkpoint at {path}")
        d = torch.load(path)
        lstm = cls.from_dict(d['dict'])
        i_start = d['epoch'] + 1
        lstm.train()
    else:

        # initialize model
        lstm = cls(
            train_data.mean,
            train_data.scale,
            train_data.timestep(),
            auxiliary=auxiliary,
            prognostic=prognostic,
            diagnostic=diagnostic,
            forcing=forcing)
        i_start = 0
        lstm.train()

    logger.info(f"Training with {lstm}")

    # initialize optimizer
    optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

    # set up loss function
    loss_keys = [x[0] for x in diagnostic + prognostic]
    criterion = MVLoss(loss_keys, constants['layer_mass'], loss_scale)

    steps_out = 0

    os.chdir(output_dir)
    try:
        for i in range(i_start, n_epochs):
            logging.info(f"Epoch {i}")
            for k, batch in enumerate(train_loader):
                steps_out += 1
                batch = batch_to_model_inputs(batch, auxiliary, prognostic,
                                              diagnostic, forcing, constants)
                logging.info(f"Batch {k} of {len(train_loader)}")
                time_batch_start = time()
                for t in range(0, nt - seq_length + 1, skip):

                    # select window
                    window = select_time(batch, slice(t, t + seq_length))
                    x = select_time(window, slice(0, -1))
                    y = select_time(window, slice(1, -1))

                    # make prediction
                    pred = lstm(x, n_prog=1)

                    # compute loss
                    loss = criterion(y, pred)

                    # Back propagate
                    optimizer.zero_grad()
                    loss.backward()

                    # take step
                    optimizer.step()

                    # Log the results
                    meter_avg_loss.add(criterion(window, mean).item())
                    meter_loss.add(loss.item())

                time_elapsed_batch = time() - time_batch_start

                batch_info = {
                    'epoch': i,
                    'batch': k,
                    'loss': meter_loss.value()[0],
                    'avg_loss': meter_avg_loss.value()[0],
                    'time_elapsed': time_elapsed_batch,
                }
                # experiment.log_metric('loss', batch_info['loss'])
                # experiment.log_metric('avg_loss', batch_info['avg_loss'])
                # db.log_batch(batch_info)

                ex.log_scalar('loss', batch_info['loss'])
                ex.log_scalar('avg_loss', batch_info['avg_loss'])
                ex.log_scalar('time_elapsed', batch_info['time_elapsed'])

                logger.info(f"Batch {k},  Loss: {meter_loss.value()[0]}; "
                            f"Avg {meter_avg_loss.value()[0]}; "
                            f"Time Elapsed {time_elapsed_batch} ")
                meter_loss.reset()
                meter_avg_loss.reset()

            # save artifacts
            if steps_out > min_output_interval:
                print("Saving")
                steps_out = 0
                epoch_file = f"{i}.pkl"
                torch.save({"epoch": i, 'dict': lstm.to_dict()}, epoch_file)
                ex.add_artifact(epoch_file)

                for artifact in after_epoch(lstm, ds, i):
                    ex.add_artifact(artifact)

    except KeyboardInterrupt:
        torch.save({'epoch': i, 'dict': lstm.to_dict()}, "interrupt.pkl")
