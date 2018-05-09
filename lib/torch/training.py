"""Fit model for the multiple time step objective function. This requires some
special dataloaders etc

"""
import json
import logging
import pprint

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .utils import train
from . import model, data


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



def train_multistep_objective(train_data, test_data, output_dir,
                              num_epochs=5,
                              num_test_examples=10000,
                              window_size={0: 2, 1: 10, 2: 20},
                              test_window_size=64,
                              num_batches=None, batch_size=200, lr=0.01,
                              weight_decay=0.0, nsteps=1, nhidden=(256,),
                              cuda=False, pytest=False,
                              precip_in_loss=False,
                              precip_positive=False,
                              radiation='zero',
                              seed=1):
    """Train a single layer perceptron euler time stepping model

    For one time step this torch models performs the following math

    .. math::

        x^n+1 = x^n + dt * (f(x^n) + g)

    Parameters
    ----------
    window_size : int or callable
        Size of prediction window for training. If this is a dict then the
        window_size of the time stepper will be set to window_size(epoch).
        Default: 10.


    """

    arguments = locals()
    arguments.pop('test_data')
    arguments.pop('train_data')
    logger.info("Called with parameters:\n" + pprint.pformat(arguments))
    logger.info(f"Saving to {output_dir}")

    json.dump(arguments, open(f"{output_dir}/arguments.json", "w"))

    torch.manual_seed(seed)

    # set window size scheduler
    default_window_size = 20
    if isinstance(window_size, int):
        default_window_size = window_size
        def window_size(epoch):
            return default_window_size
    elif isinstance(window_size, dict):
        window_size_dict = window_size
        def window_size(epoch):
            return window_size_dict.get(epoch, default_window_size)

    # the sampling interval of the data
    dt = Variable(torch.FloatTensor([3 / 24]), requires_grad=False)
    if cuda:
        dt = dt.cuda()


    test_dataset = data.to_dataset(test_data, test_window_size)
    # ntrain = len(train_dataset)
    ntest = len(test_dataset)

    # logging.info(f"Training dataset length: {ntrain}")
    logging.info(f"Testing dataset length: {ntest}")

    # define constants to be used by model
    constants = data.to_constants(train_data)
    scaler = data.to_scaler(train_data)
    loss = data.to_dynamic_loss(train_data,
                                radiation_in_loss=radiation == 'interactive', precip_in_loss=precip_in_loss,
                                cuda=cuda)


    # get testing_loader
    testing_inds = np.random.choice(len(test_dataset), num_test_examples,
                                    replace=False)
    test_loader = DataLoader(test_dataset, batch_size=len(testing_inds),
                             sampler=SubsetRandomSampler(testing_inds))


    # define the neural network
    m = data.get_num_features(train_data)
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

    # move data to gpu if appropriate
    if cuda:
        nstepper.cuda()
        for key in constants:
            constants[key] = constants[key].cuda()

    ##
    # model training code below here
    ##
    epoch_data = []

    def get_generator(epoch):
        T = window_size(epoch)
        train_dataset = data.to_dataset(train_data, T)
        training_inds = np.arange(len(train_dataset))

        # Create training data loaders
        if num_batches:
            logger.info(f"Using boostrap sample of {num_batches}.")
            training_inds = np.random.choice(len(train_dataset),
                                             num_batches * batch_size, replace=False)
        else:
            logger.info(f"Using full training dataset")

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=SubsetRandomSampler(training_inds))
        return train_loader

    def closure(batch):
        batch = _prepare_vars_in_nested_dict(batch, cuda=cuda)
        batch['constant'] = constants
        y = nstepper(batch)
        return loss(batch, y)

    def monitor(state):
        loss = sum(closure(batch) for batch in test_loader)
        avg_loss  = loss / len(test_loader)
        state['test_loss'] = float(avg_loss)
        epoch_data.append(state)
        logger.info("Epoch[batch]: {epoch}[{batch}]; Test Loss: {test_loss}; Train Loss: {train_loss}".format(**state))

    def on_epoch_start(epoch):
        file = f"{output_dir}/{epoch}/state.torch"

        logging.info(f"Begin epoch {epoch}")

        T = window_size(epoch)
        logging.info(f"Setting training window size to {T}")
        nstepper.window_size = T

        logger.info(f"Saving state to %s" % file)
        torch.save(nstepper.to_saved(), file)

    def on_finish():
        import json
        # torch.save(nstepper, f"{output_dir}/{num_epochs}/model.torch")
        torch.save(nstepper.to_saved(), f"{output_dir}/{num_epochs}/state.torch")
        json.dump(epoch_data, open(f"{output_dir}/loss.json", "w"))

    def on_error(data):
        logging.critical("Dumping data to \"dump.pt\"")
        torch.save({
            "data": data,
            "model": nstepper,
            "constants": constants
        }, "dump.pt")

    train(
        get_generator,
        closure,
        optimizer=optimizer,
        monitor=monitor,
        num_epochs=num_epochs,
        on_epoch_start=on_epoch_start,
        on_finish=on_finish,
        on_error=on_error)

    training_metadata = {
        'args': arguments,
        'training': epoch_data,
        'n_test': ntest,
        # 'n_train': ntrain
    }
    return nstepper, training_metadata
