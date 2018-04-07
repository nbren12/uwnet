from itertools import islice

import torch
from torch.autograd import Variable
import logging
from timeit import default_timer as timer



def train(data_loader, loss_fn, optimizer, num_epochs=1, monitor=None,
          on_epoch_start=None, on_finish=None):
    """Train a torch model

    Parameters
    ----------
    num_epochs : int
        number of epochs of training
    num_steps : int or None
        maximum number of batches per epoch. If None, then the full dataset is
        used.
    """
    logger = logging.getLogger(__name__)

    num_steps = len(data_loader)

    t_begin_train = timer()
    for epoch in range(num_epochs):
        avg_loss = 0
        if on_epoch_start:
            on_epoch_start(epoch)

        t_start = timer()
        for batch_idx, data in enumerate(data_loader):
            if batch_idx % 500 == 0:
                avg_loss /= num_steps
                if monitor:
                    monitor({'epoch': epoch,
                        'train_loss': float(avg_loss),
                        'batch': batch_idx})
                avg_loss = 0.0

            optimizer.zero_grad()  # this is not done automatically in torch
            # pass all data args to loss_function
            loss = loss_fn(data)
            loss.backward()
            optimizer.step()

            avg_loss += loss.data.cpu().numpy()

            if batch_idx % 200 == 99:
                t_end = timer()
                logger.debug(f"{batch_idx}/{len(data_loader)} batches done. "
                             f"Rate: {200/(t_end-t_start):.2f} batch/sec")
                t_start = timer()

    logging.info("Done Training. Time elapsed {}".format(timer()-t_begin_train))
    if on_finish:
        on_finish()
