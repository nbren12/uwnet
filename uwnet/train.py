import argparse
import logging
import os
from time import time

import yaml
from toolz import merge

import torch
import torchnet as tnt
import xarray as xr
from torch.utils.data import DataLoader
from uwnet import model
from uwnet.datasets import XRTimeSeries
from uwnet.logging import MongoDBLogger
from uwnet.loss import MVLoss
from uwnet.utils import select_time

from comet_ml import Experiment



def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', '--restart', default=False)
    parser.add_argument('-lr', '--lr', default=.001, type=float)
    parser.add_argument('-n', '--n-epochs', default=10, type=int)
    parser.add_argument('-d', '--model-dir', default='trained_models')
    parser.add_argument('-s', '--skip', default=1, type=int)
    parser.add_argument('-l', '--seq_length', default=20, type=int)
    parser.add_argument('-b', '--batch_size', default=200, type=int)
    parser.add_argument('-f', '--forcing-data', type=str, default='')
    parser.add_argument('-db', default='runs.json')
    parser.add_argument('config')
    parser.add_argument("input")

    return parser.parse_args()


def get_output_dir(run_id, base='.trained_models'):
    id = str(run_id)
    return f'{base}/{id[:2]}/{id[2:]}'


def main():

    # setup logging
    logging.basicConfig(level=logging.INFO)
    experiment = Experiment(api_key="fEusCnWmzAtmrB0FbucyEggW2")
    db = MongoDBLogger()
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    # load configuration
    config = yaml.load(open(args.config))

    # get output directory
    db.log_run(args, config)
    output_dir = get_output_dir(db.run_id, base=args.model_dir)

    experiment.log_parameter('directory', os.path.abspath(output_dir))

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    seq_length = args.seq_length

    # set up meters
    meter_loss = tnt.meter.AverageValueMeter()
    meter_avg_loss = tnt.meter.AverageValueMeter()

    # get training loader
    def post(x):
        return x

    logger.info("Opening Training Data")
    try:
        ds = xr.open_zarr(args.input)
    except ValueError:
        ds = xr.open_dataset(args.input)
    nt = len(ds.time)
    train_data = XRTimeSeries(ds.load(), [['time'], ['x', 'y'], ['z']])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    constants = train_data.torch_constants()

    # switch to output directory
    logger.info(f"Saving outputs in {output_dir}")
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    # compute standard deviation
    logger.info("Computing Standard Deviation")
    scale = train_data.scale

    # compute scaler
    logger.info("Computing Mean")
    mean = train_data.mean

    cls = model.MLP

    logger.info(f"Training with {cls}")

    # restart
    if args.restart:
        path = os.path.abspath(args.restart)
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
            inputs=config['inputs'],
            outputs=config['outputs'])
        i_start = 0

    logger.info(f"Training with {lstm}")

    # initialize optimizer
    optimizer = torch.optim.Adam(lstm.parameters(), lr=args.lr)

    os.chdir(output_dir)
    try:
        for i in range(i_start, n_epochs):
            logging.info(f"Epoch {i}")
            for k, batch in enumerate(train_loader):
                logging.info(f"Batch {k} of {len(train_loader)}")

                # set up loss function
                criterion = MVLoss(constants['layer_mass'],
                                   config['loss_scale'])

                time_batch_start = time()

                for t in range(0, nt - seq_length, args.skip):

                    # select window
                    window = select_time(batch, slice(t, t + seq_length))
                    x = select_time(window, slice(0, -1))

                    # patch the constants back in
                    x = merge(x, constants)

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
                experiment.log_metric('loss', batch_info['loss'])
                experiment.log_metric('avg_loss', batch_info['avg_loss'])
                db.log_batch(batch_info)
                logger.info(f"Batch {k},  Loss: {meter_loss.value()[0]}; "
                            f"Avg {meter_avg_loss.value()[0]}; "
                            f"Time Elapsed {time_elapsed_batch} ")
                meter_loss.reset()
                meter_avg_loss.reset()

            experiment.log_epoch_end(i)
            db.log_epoch(i, lstm)

    except KeyboardInterrupt:
        torch.save({'epoch': i, 'dict': lstm.to_dict()}, "interrupt.pkl")


if __name__ == '__main__':
    main()
