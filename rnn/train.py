import argparse
import logging

import torch
import yaml
from torch.utils.data import DataLoader

import torchnet as tnt

from . import model
from .prepare_data import get_dataset
from .utils import get_batch_size, select_time


def mse(x, y, layer_mass):
    w = layer_mass / layer_mass.mean()
    return torch.mean(torch.pow(x - y, 2) * w.float())


def criterion(x, y, layer_mass):
    return (mse(x['sl'], y['sl'], layer_mass) / scale['sl']**2 +
            mse(x['qt'], y['qt'], layer_mass) / scale['qt']**2 * 5)


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', '--restart', default=False)
    parser.add_argument('-lr', '--lr', default=.001, type=float)
    parser.add_argument('-n', '--n-epochs', default=10, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    # setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # load configuration
    config = yaml.load(open("config.yaml"))

    args = parse_arguments()

    n_epochs = args.n_epochs
    batch_size = 100
    seq_length = 20
    skip = 10
    nt = 640

    # set up meters
    meter_loss = tnt.meter.AverageValueMeter()
    meter_avg_loss = tnt.meter.AverageValueMeter()

    # open training data
    paths = config['paths']

    # get training loader
    def post(x):
        return x.isel(y=slice(24, 40))

    train_data = get_dataset(paths, post=post)
    train_loader = DataLoader(train_data, batch_size=batch_size)

    logger.info("Computing Standard Deviation")
    scale = train_data.scale
    # compute scaler
    logger.info("Computing Mean")
    mean = train_data.mean

    cls = model.MLP

    logger.info(f"Training with {cls}")

    # restart
    if args.restart:
        logger.info(f"Restarting from checkpoint at {args.restart}")
        d = torch.load(args.restart)
        lstm = cls.from_dict(d['dict'])
        i_start = d['epoch'] + 1
    else:

        # initialize model
        lstm = cls(mean, scale)
        i_start = 0

    # initialize optimizer
    optimizer = torch.optim.Adam(lstm.parameters(), lr=args.lr)

    try:
        for i in range(i_start, n_epochs):
            logging.info(f"Epoch {i}")
            for k, batch in enumerate(train_loader):
                logging.info(f"Batch {k} of {len(train_loader)}")
                n = get_batch_size(batch)

                # need to remove these auxiliary variables
                batch.pop('p')
                dm = batch.pop('layer_mass').detach_()
                loss = 0.0

                for t in range(0, nt - seq_length, skip):
                    window = select_time(batch, slice(t, t + seq_length))
                    pred = lstm(window, n=1)
                    loss = criterion(pred, window, dm[0, :])

                    # average
                    meter_avg_loss.add(
                        criterion(mean, window, dm[0, :]).item())
                    meter_loss.add(loss.item())

                    # take step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # try:
                    #     for x in hid:
                    #         x.detach_()

                    # except TypeError:
                    #     pass

                    # loss = 0.0

                logger.info(f"Batch {k},  Loss: {meter_loss.value()[0]};"
                            f" Avg {meter_avg_loss.value()[0]}")
                meter_loss.reset()
                meter_avg_loss.reset()

            logger.info(f"Saving checkpoint to {i}.pkl")
            torch.save({'epoch': i, 'dict': lstm.to_dict()}, f"{i}.pkl")

    except KeyboardInterrupt:
        torch.save({'epoch': i, 'dict': lstm.to_dict()}, "interrupt.pkl")
