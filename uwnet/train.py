import logging
import os
import re
from os.path import join
from time import time

from sacred import Experiment
from toolz import merge

import torch
import torchnet as tnt
import xarray as xr
from torch.utils.data import DataLoader
from uwnet import model
from uwnet.datasets import XRTimeSeries
from uwnet.loss import MVLoss
from uwnet.utils import select_time, batch_to_model_inputs

ex = Experiment()


def get_output_dir(tag=None, base='.trained_models'):
    """Get a unique output directory name"""

    pat = re.compile(r'^(\d+)-?')
    files = os.listdir(base)

    if len(files) == 0:
        id = '0'
    else:
        last_id = max(int(pat.search(file).group(1)) for file in files)
        id = last_id + 1

    if not tag:
        file_name = str(id)
    else:
        if re.search(r'\s', tag):
            raise ValueError("Tag has whitespace")
        file_name = str(id) + '-' + tag

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


@ex.automain
def main(inputs, forcings, outputs, restart, lr, n_epochs, model_dir, skip, seq_length,
         batch_size, tag, data, vertical_grid_size, loss_scale):
    # setup logging
    logging.basicConfig(level=logging.INFO)

    # db = MongoDBLogger()
    # experiment = Experiment(api_key="fEusCnWmzAtmrB0FbucyEggW2")
    logger = logging.getLogger(__name__)

    # get output directory
    # db.log_run(args, config)
    output_dir = get_output_dir(tag, base=model_dir)
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

    nt = len(ds.time)
    ds = ds.isel(z=slice(0, vertical_grid_size))
    train_data = XRTimeSeries(ds.load(), [['time'], ['x', 'y'], ['z']])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    constants = train_data.torch_constants()

    # compute standard deviation
    logger.info("Computing Standard Deviation")
    scale = train_data.scale

    # compute scaler
    logger.info("Computing Mean")
    mean = train_data.mean

    cls = model.MLP

    logger.info(f"Training with {cls}")

    # restart
    if restart:
        path = os.path.abspath(restart)
        logger.info(f"Restarting from checkpoint at {path}")
        d = torch.load(path)
        lstm = cls.from_dict(d['dict'])
        i_start = d['epoch'] + 1
    else:

        # initialize model
        lstm = cls(
            mean,
            scale,
            time_step=train_data.timestep(),
            inputs=inputs,
            forcings=forcings,
            outputs=outputs)
        i_start = 0
        lstm.train()

    logger.info(f"Training with {lstm}")

    # initialize optimizer
    optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

    # set up loss function
    criterion = MVLoss(lstm.outputs.names, constants['layer_mass'], loss_scale)

    os.chdir(output_dir)
    try:
        for i in range(i_start, n_epochs):
            logging.info(f"Epoch {i}")
            for k, batch in enumerate(train_loader):
                batch = batch_to_model_inputs(
                    batch, inputs, forcings, outputs, constants)
                logging.info(f"Batch {k} of {len(train_loader)}")
                time_batch_start = time()
                for t in range(0, nt - seq_length, skip):

                    # select window
                    window = select_time(batch, slice(t, t + seq_length))
                    x = select_time(window, slice(0, -1))
                    y = select_time(window, slice(1, None))

                    # make prediction
                    pred = lstm(x, n=1)

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

            epoch_file = f"{i}.pkl"
            torch.save({"epoch": i, 'dict': lstm.to_dict()}, epoch_file)
            ex.add_artifact(epoch_file)

    except KeyboardInterrupt:
        torch.save({'epoch': i, 'dict': lstm.to_dict()}, "interrupt.pkl")
