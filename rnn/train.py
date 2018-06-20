import logging

import yaml

import torch
from torch import nn
from toolz import merge_with

from torch.utils.data import DataLoader
from lib.torch.normalization import scaler
from .prepare_data import get_dataset
from .model import SimpleLSTM


def select_time(batch, i):
    out = {}
    for key, val in batch.items():
        if val.dim() == 1:
            out[key] = val
        else:
            out[key] = val[:,i]
    return out


def get_batch_size(batch):
    return batch['sl'].size(0)


def concat_dicts(seq):
    return merge_with(lambda x: torch.cat(seq, dim=1),
                      *seq)


def mse(x, y, layer_mass):
    w = torch.sqrt(layer_mass).float()
    return nn.MSELoss()(x*w, y*w)


def criterion(x, y, layer_mass):
    return  (
        mse(x['sl'], y['sl'], layer_mass)/scale['sl']**2
        + mse(x['qt'], y['qt'], layer_mass)/scale['qt']**2
             )


if __name__ == '__main__':
    # setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # load configuration
    config = yaml.load(open("config.yaml"))

    n_epochs = 5
    batch_size = 100
    seq_length = 100
    nt = 640


    # open training data
    paths = config['paths']

    # get training loader
    train_data = get_dataset(paths)
    train_loader = DataLoader(train_data, batch_size=batch_size)

    # compute scaler
    logger.info("Computing Mean")
    mean = train_data.mean
    logger.info("Computing Standard Deviation")
    scale = train_data.scale
    scaler = scaler(scale, mean)

    # initialize model
    lstm = SimpleLSTM(scaler)

    # initialize optimizer
    optimizer = torch.optim.Adam(lstm.parameters(), lr=.01)

    for i in range(n_epochs):
        logging.info(f"Epoch {i}")
        for k, batch in enumerate(train_loader):
            n = get_batch_size(batch)
            hid = lstm.init_hidden(n)

            # need to remove these auxiliary variables
            batch.pop('p')
            dm = batch.pop('layer_mass').detach_()

            loss = 0.0
            for t in range(nt-1):
                pred, hid = lstm(select_time(batch, t), hid)
                loss += criterion(pred, select_time(batch, t+1), dm[0,:])

                if t % seq_length == 0:
                    print(t, float(loss/seq_length))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    for x in hid:
                        x.detach_()
                    loss = 0.0


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
