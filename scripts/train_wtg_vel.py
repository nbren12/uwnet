import xarray as xr
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
train_loss_logger = VisdomPlotLogger(
    'line', port=8097, opts={'title': 'Train Loss'})


from lib.models.torch.torch_datasets import ConcatDataset
from lib.models.torch.torch_models import train
from lib.models.torch.preprocess import prepare_data


class Prediction(nn.Module):
    def __init__(self):
        super(Prediction, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm1d(54),
            nn.Linear(54, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.ReLU(),
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 34),
        )


    def forward(self, x):
        return self.net(x)

def _get_data_loader(ds):
    nz = ds.z.shape[0]
    dim_order = ['time', 'y', 'x', 'z']

    data ={key: ds[key].transpose(*dim_order).values.reshape((-1, nz))
           for key in ds
           if set(ds[key].dims) == set(dim_order)}

    X = np.concatenate((data['qt'][:,:-14],
                        data['sl']), axis=-1)
    Y = data['W']

    return DataLoader(ConcatDataset(X, Y), batch_size=100)

def score(net, data_loader):
    x, g = [torch.cat(x, 0) for x in zip(*iter(data_loader))]
    net.eval()

    g_pred = net.forward(Variable(x).float()).data.numpy()
    g = g.numpy()

    ss = np.power(g-g.mean(0), 2).sum(0)
    sse = np.power(g_pred - g, 2).sum(0)

    return 1 - sse/ss

torch.manual_seed(1)

files = [
    "data/raw/ngaqua/coarse/3d/W.destaggered.nc",
    "data/calc/ngaqua/qt.nc",
    "data/calc/ngaqua/sl.nc"
]

def preprocess(x):
    if 'p' in x:
        x = x.drop('p')
    return x.isel(y=slice(24,40))

ds = xr.open_mfdataset(files, preprocess=preprocess)
data_loader = _get_data_loader(ds.isel(time=slice(0,-200)))
test_loader = _get_data_loader(ds.isel(time=slice(-200,None)))

net = Prediction()
optimizer = torch.optim.Adam(net.parameters(), lr=.0001)
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

    # try:
    #     train_loss_logger.log(counter, loss_run_avg)
    # except:
    #     pass

    return loss


train(data_loader, loss, optimizer, num_epochs=2)

R2 = score(net, test_loader)
print(R2)
