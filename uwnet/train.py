import argparse
import logging
import os

import torch
import torchnet as tnt
import yaml
from toolz import curry, merge
from time import time
from torch.utils.data import DataLoader

import xarray as xr
from uwnet import model
from uwnet.data import XRTimeSeries
from uwnet.utils import select_time


def get_git_rev():
    import subprocess
    import uwnet
    root_dir = os.path.dirname(uwnet.__path__[0])
    out = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root_dir)
    return out.decode().strip()


def mse(x, y, layer_mass):
    x = x.float()
    y = y.float()
    layer_mass = layer_mass.float()
    w = layer_mass / layer_mass.mean()

    if x.dim() == 2:
        x = x[..., None]

    if x.size(-1) > 1:
        if layer_mass.size(-1) != x.size(-1):
            raise ValueError

        return torch.mean(torch.pow(x - y, 2) * w)
    else:
        return torch.mean(torch.pow(x - y, 2))


@curry
def MVLoss(layer_mass, scale, x, y):
    """MSE loss

    Parameters
    ----------
    x : truth
    y : prediction
    """

    losses = {
        key:
        mse(x[key], y[key], layer_mass) / torch.tensor(scale[key]**2).float()
        for key in scale
    }
    return sum(losses.values())


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', '--restart', default=False)
    parser.add_argument('-lr', '--lr', default=.001, type=float)
    parser.add_argument('-n', '--n-epochs', default=10, type=int)
    parser.add_argument('-o', '--output-dir', default='.')
    parser.add_argument('-s', '--skip', default=1, type=int)
    parser.add_argument('-l', '--seq_length', default=20, type=int)
    parser.add_argument('-b', '--batch_size', default=200, type=int)
    parser.add_argument('-db', default='runs.json')
    parser.add_argument('config')
    parser.add_argument("input")

    return parser.parse_args()


def main():

    # setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    # load configuration
    config = yaml.load(open(args.config))

    import tinydb

    # open up tinydb
    db = tinydb.TinyDB(args.db)
    log_table = db.table('batches')
    run_table = db.table('runs')
    epoch_table = db.table('epochs')

    output_dir = os.path.abspath(args.output_dir)

    run_id = run_table.insert({
        "run": output_dir,
        'training_data': os.path.abspath(args.input),
        "config": config,
        "args": vars(args),
        'git': {
            'rev': get_git_rev()
        }
    })

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    seq_length = args.seq_length
    nt = 640

    # set up meters
    meter_loss = tnt.meter.AverageValueMeter()
    meter_avg_loss = tnt.meter.AverageValueMeter()

    # get training loader
    def post(x):
        return x

    logger.info("Opening Training Data")
    ds = xr.open_zarr(args.input)
    train_data = XRTimeSeries(ds.load(), [['time'], ['x', 'y'], ['z']])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    constants = train_data.torch_constants()

    # switch to output directory
    logger.info(f"Saving outputs in {args.output_dir}")
    try:
        os.mkdir(output_dir)
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
                    'run': run_id,
                    'epoch': i,
                    'batch': k,
                    'loss': meter_loss.value()[0],
                    'avg_loss': meter_avg_loss.value()[0],
                    'time_elapsed': time_elapsed_batch,
                }
                log_table.insert(batch_info)

                logger.info(f"Batch {k},  Loss: {meter_loss.value()[0]}; "
                            f"Avg {meter_avg_loss.value()[0]}; "
                            f"Time Elapsed {time_elapsed_batch} ")
                meter_loss.reset()
                meter_avg_loss.reset()

            logger.info(f"Saving checkpoint to {i}.pkl")
            torch.save({'epoch': i, 'dict': lstm.to_dict()}, f"{i}.pkl")
            epoch_table.insert({
                'model': os.path.join(output_dir, f'{i}.pkl'),
                'run': run_id
            })

    except KeyboardInterrupt:
        torch.save({'epoch': i, 'dict': lstm.to_dict()}, "interrupt.pkl")


if __name__ == '__main__':
    main()
