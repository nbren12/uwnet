from itertools import islice

import torch
from torch.autograd import Variable
from tqdm import tqdm


def train(data_loader, loss_fn, optimizer, num_epochs=1, monitor=None):
    """Train a torch model

    Parameters
    ----------
    num_epochs : int
        number of epochs of training
    num_steps : int or None
        maximum number of batches per epoch. If None, then the full dataset is
        used.
    """

    num_steps = len(data_loader)

    if monitor:
        train_loss = sum(loss_fn(batch) for batch in data_loader)/len(data_loader)
        monitor({'epoch': -1, 'train_loss': float(train_loss)})

    for epoch in range(num_epochs):
        avg_loss = 0
        for batch_idx, data in tqdm(enumerate(data_loader), total=num_steps):
            optimizer.zero_grad()  # this is not done automatically in torch

            # pass all data args to loss_function
            loss = loss_fn(data)

            loss.backward()
            optimizer.step()

            avg_loss += loss.data.cpu().numpy()

        avg_loss /= num_steps

        if monitor:
            monitor({'epoch': epoch, 'train_loss': float(avg_loss)})



def jacobian(net, x):
    """Compute the Jacobian of a torch module with respect to its inputs"""
    x0 = torch.FloatTensor(np.squeeze(x)).double()
    x0 = Variable(x0, requires_grad=True)

    nv = x0.size(0)

    jac = torch.zeros(nv, nv)

    for i in range(nv):
        outi = net(x0)[i]
        outi.backward()
        jac[i, :] = x0.grad.data
        x0.grad.data.zero_()

    return jac.numpy()
