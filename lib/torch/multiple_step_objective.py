"""Fit model for the multiple time step objective function. This requires some
special dataloaders etc

"""
import json
import logging
import pprint

import numpy as np
import torch
from toolz import curry, merge_with, valmap
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .datasets import DictDataset, WindowedData
from .utils import train
from . import model

logger = logging.getLogger(__name__)


def _prepare_vars_in_nested_dict(data, cuda=False):
    if torch.is_tensor(data):
        x = Variable(data).float()
        if cuda:
            x = x.cuda()
        return x.transpose(0, 1)
    elif isinstance(data, dict):
        return {
            key: _prepare_vars_in_nested_dict(val, cuda=cuda)
            for key, val in data.items()
        }


def prepare_dataset(data,
                    window_size,
                    prognostic_variables=('sl', 'qt'),
                    forcing_variables=('sl', 'qt', 'QRAD', 'LHF', 'SHF',
                                       'Prec', 'W', 'SOLIN')):


    X, G = data['X'], data['G']
    X = DictDataset({
        key: WindowedData(X[key], window_size)
        for key in prognostic_variables
    })
    G = DictDataset(
        {key: WindowedData(G[key], window_size)
         for key in forcing_variables})

    return DictDataset({'prognostic': X, 'forcing': G})


def _numpy_to_variable(x):
    return Variable(torch.FloatTensor(x))


def _data_to_scaler(data, cuda=False):
    # compute mean and stddev
    # this is an error, std does not work like this
    means = {}
    scales = {}
    for key in data['X']:
        X = data['X'][key]
        m = X.shape[-1]
        mu = X.reshape((-1, m)).mean(axis=0)
        sig = X.reshape((-1, m)).std(axis=0)

        # convert to torch
        mu, sig = [_numpy_to_variable(np.squeeze(x)) for x in [mu, sig]]
        sig = torch.mean(sig)

        if cuda:
            mu = mu.cuda()
            sig = sig.cuda()

        means[key] = mu
        scales[key] = sig

    return scaler(scales, means)


def _data_to_loss_feature_weights(data, cuda=True):
    def _f(args):
        w, scale = args
        return w / scale**2

    w = merge_with(_f, data['w'], data['scales'])
    w = valmap(_numpy_to_variable, w)

    if cuda:
        w = valmap(lambda x: x.cuda(), w)

    return w


def _scale_var(scale, mean, x):
    x = x.double()
    mu = mean.double()
    sig = scale.double()

    x = x.sub(mu)
    x = x.div(sig + 1e-7)

    return x.float()


@curry
def scaler(scales, means, x):
    out = {}
    for key in x:
        if key in scales and key in means:
            out[key] = _scale_var(scales[key], means[key], x[key])
        else:
            out[key] = x[key]
    return out


@curry
def weighted_loss(weight, x, y):
    # return torch.mean(torch.pow(x - y, 2).mul(weight.float()))
    return torch.mean(torch.abs(x - y).mul(weight.float()))


def train_multistep_objective(train_data, test_data, output_dir,
                              num_epochs=5,
                              num_test_examples=10000,
                              window_size=10,
                              test_window_size=64,
                              num_batches=None, batch_size=200, lr=0.01,
                              weight_decay=0.0, nsteps=1, nhidden=(128,),
                              cuda=False, pytest=False,
                              precip_in_loss=False,
                              precip_positive=False,
                              radiation='zero',
                              seed=1):
    """Train a single layer perceptron euler time stepping model

    For one time step this torch models performs the following math

    .. math::

        x^n+1 = x^n + dt * (f(x^n) + g)


    """

    arguments = locals()
    arguments.pop('test_data')
    arguments.pop('train_data')
    logger.info("Called with parameters:\n" + pprint.pformat(arguments))
    logger.info(f"Saving to {output_dir}")

    json.dump(arguments, open(f"{output_dir}/arguments.json", "w"))

    torch.manual_seed(seed)

    # the sampling interval of the data
    dt = Variable(torch.FloatTensor([3 / 24]), requires_grad=False)
    if cuda:
        dt = dt.cuda()

    train_slice = slice(200, None)
    test_slice = slice(0, 200)

    train_dataset = prepare_dataset(train_data, window_size)
    test_dataset = prepare_dataset(test_data, test_window_size)

    ntrain = len(train_dataset)
    ntest = len(test_dataset)
    logging.info(f"Training dataset length: {ntrain}")
    logging.info(f"Testing dataset length: {ntest}")

    # train on only a bootstrap sample
    if num_batches:
        logger.info(f"Using boostrap sample of {num_batches}. Setting "
                    "number of epochs to 1.")
        num_epochs = 1
        training_inds = np.random.choice(len(train_dataset),
                                         num_batches * batch_size, replace=False)
    else:
        logger.info(f"Using full training dataset")
        training_inds = np.arange(len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=SubsetRandomSampler(training_inds))

    # make a test_loader object
    testing_inds = np.random.choice(len(test_dataset), num_test_examples,
                                    replace=False)
    test_loader = DataLoader(test_dataset, batch_size=len(testing_inds),
                             sampler=SubsetRandomSampler(testing_inds))

    scaler = _data_to_scaler(train_data, cuda=cuda)
    weights = _data_to_loss_feature_weights(train_data, cuda=cuda)

    # define the neural network
    m = sum(valmap(lambda x: x.size(-1), weights).values())

    rhs = model.RHS(
        m,
        hidden=nhidden,
        scaler=scaler,
        radiation=radiation,
        precip_positive=precip_positive)

    nstepper = model.ForcedStepper(
        rhs,
        h=dt,
        nsteps=nsteps)

    optimizer = torch.optim.Adam(
        rhs.parameters(), lr=lr, weight_decay=weight_decay)

    constants = {
        'w': Variable(torch.FloatTensor(train_data['w']['sl'])),
        'z': Variable(torch.FloatTensor(train_data['z']))
    }
    if cuda:
        nstepper.cuda()
        for key in constants:
            constants[key] = constants[key].cuda()

    def loss(truth, pred):
        x = truth['prognostic']
        y = pred['prognostic']

        total_loss = 0
        # time series loss
        for key in y:
            total_loss += weighted_loss(weights[key], x[key], y[key]) / len(y)

        if precip_in_loss:
            # column budget losses this compares he predicted precipitation for
            # each field to reality
            prec = truth['forcing']['Prec']
            predicted = pred['diagnostic']['Prec']
            observed = (prec[1:] + prec[:-1]) / 2
            total_loss += torch.mean(torch.pow(observed - predicted, 2)) / 5

        if radiation == 'interactive':
            qrad = truth['forcing']['QRAD']
            predicted = pred['diagnostic']['QRAD'][0]
            observed = qrad[0]
            total_loss += torch.mean(torch.pow(observed - predicted, 2))

        return total_loss

    # _init_linear_weights(net, .01/nsteps)
    def closure(batch):
        batch = _prepare_vars_in_nested_dict(batch, cuda=cuda)
        batch['constant'] = constants
        y = nstepper(batch)
        return loss(batch, y)

    epoch_data = []

    def monitor(state):
        loss = sum(closure(batch) for batch in test_loader)
        avg_loss  = loss / len(test_loader)
        state['test_loss'] = float(avg_loss)
        epoch_data.append(state)
        logger.info("Epoch[batch]: {epoch}[{batch}]; Test Loss: {test_loss}; Train Loss: {train_loss}".format(**state))

    def on_epoch_start(epoch):
        torch.save(nstepper, f"{output_dir}/{epoch}/model.torch")

    def on_finish():
        import json
        torch.save(nstepper, f"{output_dir}/{num_epochs}/model.torch")
        json.dump(epoch_data, open(f"{output_dir}/loss.json", "w"))

    train(
        train_loader,
        closure,
        optimizer=optimizer,
        monitor=monitor,
        num_epochs=num_epochs,
        on_epoch_start=on_epoch_start,
        on_finish=on_finish)

    training_metadata = {
        'args': arguments,
        'training': epoch_data,
        'n_test': ntest,
        'n_train': ntrain
    }
    return nstepper, training_metadata
