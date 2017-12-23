import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
train_loss_logger = VisdomPlotLogger(
    'line', port=8097, opts={'title': 'Train Loss'})


from lib.models.torch.torch_datasets import WindowedData
from lib.models.torch.torch_models import train
from lib.models.torch.preprocess import prepare_data



def _get_data_loader():
    data = prepare_data("data/prog/*.nc", "data/forcing/*.nc", "data/w.nc",
                        subset_fn=lambda x: x)

    from IPython import embed; embed()
    X = data['X'].reshape((-1, 40))
    G = data['G'].reshape((-1, 40))

    return DataLoader(ConcatDataset(X, G), batch_size=100), X, G

def score(net, data_loader):
    x, g = [torch.cat(x, 0) for x in zip(*iter(data_loader))]
    net.eval()


    ss = np.power(g-g.mean(0), 2).sum(0)
    sse = np.power(g_pred - g, 2).sum(0)

    return 1 - sse/ss

data_loader, X, G = _get_data_loader()
optimizer = torch.optim.Adam(net.parameters(), lr=.01)
mse_loss = nn.MSELoss()

counter = 0
def loss(x, g):
    global counter, loss_run_avg
    x = Variable(x).float()
    g = Variable(g).float()

    loss =  mse_loss(net(x), g)

    counter += 1

    try:
        loss_run_avg
    except NameError:
        loss_run_avg = loss.data[0]
    else:
        loss_run_avg = loss_run_avg * .99 + .01 * loss.data[0]
    train_loss_logger.log(counter, loss_run_avg)

    return loss


train(data_loader, loss, optimizer, num_epochs=6)

R2 = score(net, data_loader)
print(R2)
