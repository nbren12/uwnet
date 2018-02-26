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

    for epoch in range(num_epochs):
        avg_loss = 0
        counter = 0
        for batch_idx, data in tqdm(enumerate(data_loader), total=num_steps):
            optimizer.zero_grad()  # this is not done automatically in torch

            # pass all data args to loss_function
            loss = loss_fn(data)

            loss.backward()
            optimizer.step()

            avg_loss += loss.data.cpu().numpy()
            counter += 1

            if monitor:
                monitor({'batch': batch_idx, 'epoch': epoch})

        avg_loss /= counter
        print(f"Epoch: {epoch} [{batch_idx}]\tLoss: {avg_loss}")


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
