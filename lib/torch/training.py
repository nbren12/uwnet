"""Fit model for the multiple time step objective function. This requires some
special dataloaders etc

"""
import os
import json
import logging
import pprint

import torch
from torch.autograd import Variable

from .utils import train
from . import model


logger = logging.getLogger(__name__)


def save(model, path):
    """Save object using torch.save, creating any directories in the path"""
    try:
        os.mkdir(os.path.dirname(path))
    except OSError:
        pass

    torch.save(model, path)


def train_multistep_objective(train_data, test_data, output_dir,
                              num_epochs=5,
                              num_test_examples=10000,
                              window_size={0: 2, 1: 10, 2: 20},
                              test_window_size=64,
                              num_samples=None, batch_size=200, lr=0.01,
                              weight_decay=0.0, nsteps=1, nhidden=(256,),
                              cuda=False, pytest=False,
                              precip_positive=False,
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

    try:
        os.mkdir(output_dir)
    except OSError:
        pass


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


    test_dataset = test_data.torch_dataset(test_window_size)
    # ntrain = len(train_dataset)
    ntest = len(test_dataset)

    # logging.info(f"Training dataset length: {ntrain}")
    logging.info(f"Testing dataset length: {ntest}")

    # define constants to be used by model
    scaler = train_data.scaler()
    loss = train_data.dynamic_loss()


    # get testing_loader
    test_loader = test_data.get_loader(test_window_size,
                                       num_samples=num_test_examples)

    # define the neural network
    m = train_data.get_num_features()
    rhs = model.RHS(
        m,
        hidden=nhidden,
        scaler=scaler,
        precip_positive=precip_positive)

    nstepper = model.ForcedStepper(
        rhs,
        h=dt,
        nsteps=nsteps)

    optimizer = torch.optim.Adam(
        rhs.parameters(), lr=lr, weight_decay=weight_decay)


    ##
    # model training code below here
    ##
    epoch_data = []

    def get_generator(epoch):
        T = window_size(epoch)
        return train_data.get_loader(T, num_samples=num_samples,
                                     batch_size=batch_size, shuffle=True)

    def closure(batch):
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

        save(nstepper.to_saved(), file)

    def on_finish():
        import json
        # torch.save(nstepper, f"{output_dir}/{num_epochs}/model.torch")
        save(nstepper.to_saved(), f"{output_dir}/{num_epochs}/state.torch")
        json.dump(epoch_data, open(f"{output_dir}/loss.json", "w"))

    def on_error(data):
        logging.critical("Dumping data to \"dump.pt\"")
        save({
            "data": data,
            "model": nstepper,
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
